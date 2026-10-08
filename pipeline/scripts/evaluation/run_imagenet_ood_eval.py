"""ImageNet OOD classification accuracy for the three timm architectures (RN50, ViT-B/16, ConvNeXt-B).

These models were trained with 1000-class ImageNet classification heads, so ImageNet
OOD variants (imagenet-a, imagenet-o, imagenet-r, imagenet_sketch, imagenetv2, objectnet)
can be evaluated directly — no linear probe needed.

For each (run, epoch): load the full timm classifier, forward-pass the OOD test set,
select only the logit columns present in the variant's class subset, top-1 + top-5.

CLIP ViT-B/32 is excluded: it has no classification head (contrastive training).

Usage
-----
    python pipeline/scripts/evaluation/run_imagenet_ood_eval.py            # all three architectures
    python pipeline/scripts/evaluation/run_imagenet_ood_eval.py --architecture RN50 --runs baseline
    python pipeline/scripts/evaluation/run_imagenet_ood_eval.py --architecture ViT-B/16 --datasets imagenetv2
    python pipeline/scripts/evaluation/run_imagenet_ood_eval.py --dry-run --architecture ConvNeXt-B

Results: `<results_root>/imagenet_ood_eval_scratch/<run>/imagenet_ood_eval_epoch{E}_{tag}.json`.
OOD datasets are read from the `probe_datasets` root.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths  # noqa: E402
from temporal_perturbations.data.eval_transforms import eval_preprocessing  # noqa: E402
from temporal_perturbations.models.scratch.scratch_utils import (  # noqa: E402
    ARCH_PREFIX,
    _TIMM_ARCH_MAP,
    available_scratch_epochs,
    load_timm_state_dict_from_lightning,
    scratch_checkpoint_at_epoch,
)
from temporal_perturbations.utils.run_hparams import onset_epoch as _onset_epoch  # noqa: E402

TIMM_ARCHITECTURES = ["RN50", "ViT-B/16", "ConvNeXt-B"]

DEFAULT_OUTPUT_DIR = paths.results_root() / "imagenet_ood_eval_scratch"

OOD_DATASETS = [
    "imagenet-a",
    "imagenet-o",
    "imagenet-r",
    "imagenet_sketch",
    "imagenetv2",
]

ALL_IMAGENET_WNIDS = None


def _get_all_imagenet_wnids():
    global ALL_IMAGENET_WNIDS
    if ALL_IMAGENET_WNIDS is None:
        from clip_benchmark.datasets.builder import all_imagenet_wordnet_ids
        ALL_IMAGENET_WNIDS = list(all_imagenet_wordnet_ids)
    return ALL_IMAGENET_WNIDS


def _build_ood_dataset(dataset_name, transform, data_root, download=True):
    """Build a test-only OOD ImageNet variant dataset.

    Returns (dataset, class_mask) where class_mask is a boolean array of length 1000
    indicating which ImageNet classes are present in this variant.

    Synset-dir datasets (imagenet-a/o/r/sketch) use torchvision ImageFolder directly.
    ImageNetV2 (integer dirs) uses clip_benchmark for correct label mapping.
    """
    from torchvision.datasets import ImageFolder

    all_wnids = _get_all_imagenet_wnids()
    wnid_to_idx = {w: i for i, w in enumerate(all_wnids)}

    if dataset_name == "imagenetv2":
        from clip_benchmark.datasets.builder import build_dataset
        ds = build_dataset(
            dataset_name=dataset_name, transform=transform,
            split="test", root=str(data_root), download=download,
        )
        return ds, np.ones(1000, dtype=bool)

    ds_path = data_root / dataset_name
    if not ds_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {ds_path}")

    ds = ImageFolder(str(ds_path), transform=transform)

    ds_wnids = sorted(ds.class_to_idx.keys())
    variant_wnid_set = set(ds_wnids)
    mask = np.array([w in variant_wnid_set for w in all_wnids])

    return ds, mask


def _classify_ood(model, dataset, class_mask, device, batch_size, num_workers, logger):
    """Run classification on an OOD dataset using the model's trained head.

    The model outputs 1000-class logits. For datasets with a class subset
    (imagenet-a/o/r), we select only the logit columns in that subset before
    taking argmax. The dataset's labels are positional (0..N_subset-1), matching
    the order of True entries in class_mask.
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)

    mask_indices = np.where(class_mask)[0]
    n_classes = int(class_mask.sum())

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)

            if n_classes < 1000:
                logits = logits[:, mask_indices]

            _, pred_top1 = logits.topk(1, dim=1)
            pred_top1 = pred_top1.squeeze(1)
            labels_dev = labels.to(device)
            correct_top1 += (pred_top1 == labels_dev).sum().item()

            k = min(5, n_classes)
            _, pred_topk = logits.topk(k, dim=1)
            correct_top5 += (pred_topk == labels_dev.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    top1 = correct_top1 / total
    top5 = correct_top5 / total if n_classes > 5 else None
    return top1, top5, total


def discover_runs(runs_root, architecture, logger):
    """Find all runs for an architecture, returning (baseline_dir, run_list)."""
    prefix = ARCH_PREFIX[architecture]
    baseline_name = f"{prefix}_baseline"
    baseline_dir = runs_root / baseline_name

    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline not found: {baseline_dir}")

    runs = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if d.name == baseline_name:
            onset = None
            is_baseline = True
        else:
            is_baseline = False
            try:
                onset = _onset_epoch(d)
            except Exception:
                logger.debug("  skipping %s (no onset epoch)", d.name)
                continue

        ckpt_dir = d / "checkpoints" / "all_epochs"
        if not ckpt_dir.is_dir():
            continue

        runs.append({
            "name": d.name,
            "dir": d,
            "onset": onset,
            "is_baseline": is_baseline,
        })

    return baseline_dir, runs


def build_timm_classifier(architecture, device):
    """Build the full timm model (with classification head) for direct inference."""
    import timm
    net = timm.create_model(_TIMM_ARCH_MAP[architecture], pretrained=False, num_classes=1000)
    net = net.to(device)
    net.eval()

    def load_weights(ckpt_path):
        sd = load_timm_state_dict_from_lightning(str(ckpt_path), map_location="cpu")
        net.load_state_dict(sd, strict=True)
        net.eval()
        del sd
        gc.collect()
        return net

    return net, load_weights


def _select_epochs(run_info, avail, args, all_perturbed_onsets):
    if args.all_epochs:
        return avail
    if args.onset_and_final:
        epochs = set()
        if run_info["is_baseline"]:
            epochs.update(all_perturbed_onsets)
        elif run_info["onset"] is not None and run_info["onset"] > 0:
            epochs.add(run_info["onset"])
            epochs.add(run_info["onset"] + 1)
            epochs.add(run_info["onset"] + 2)
        if avail:
            epochs.add(avail[-1])
        return sorted(e for e in epochs if e in set(avail))
    if args.epochs:
        return [int(e) for e in args.epochs.split(",")]
    return [avail[-1]] if avail else []


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--architecture", default=None, choices=TIMM_ARCHITECTURES,
                   help="Default: all three, one subprocess each.")
    p.add_argument("--runs_root", default=None, help="Default: paths key runs_root.")
    p.add_argument("--runs", default="all", choices=["all", "baseline", "perturbed"])
    p.add_argument("--datasets", default=",".join(OOD_DATASETS))
    p.add_argument("--data_root", default=None, help="Default: paths key probe_datasets.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epochs", default=None,
                   help="Comma-separated epochs to evaluate. Default: final epoch.")
    p.add_argument("--all_epochs", action="store_true",
                   help="Evaluate every available checkpoint epoch per run.")
    p.add_argument("--onset_and_final", action="store_true",
                   help="Evaluate onset epoch + final epoch per perturbed run, "
                        "and all referenced onsets + final for baselines.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.architecture is None:
        failed = [a for a in TIMM_ARCHITECTURES
                  if subprocess.run([sys.executable, __file__, "--architecture", a,
                                     *sys.argv[1:]]).returncode]
        if failed:
            sys.exit(f"failed: {failed}")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("imagenet_ood_eval")

    runs_root = Path(args.runs_root or paths.path("runs_root"))
    data_root = Path(args.data_root or paths.path("probe_datasets"))
    device = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    download = not args.no_download

    preprocessing = eval_preprocessing(args.architecture)
    transform = preprocessing.build()

    baseline_dir, runs = discover_runs(runs_root, args.architecture, logger)

    if args.runs == "baseline":
        runs = [r for r in runs if r["is_baseline"]]
    elif args.runs == "perturbed":
        runs = [r for r in runs if not r["is_baseline"]]

    logger.info("[sweep] architecture=%s runs=%d datasets=%s",
                args.architecture, len(runs), datasets)

    all_perturbed_onsets = set()
    if args.onset_and_final:
        for r in runs:
            if not r["is_baseline"] and r["onset"] is not None and r["onset"] > 0:
                all_perturbed_onsets.add(r["onset"])
                all_perturbed_onsets.add(r["onset"] + 1)
                all_perturbed_onsets.add(r["onset"] + 2)
        logger.info("[onset_and_final] baseline will be evaluated at onsets+post %s + final",
                     sorted(all_perturbed_onsets))

    if args.dry_run:
        for r in runs:
            avail = sorted(available_scratch_epochs(r["dir"]))
            final = avail[-1] if avail else "?"
            epochs_sel = _select_epochs(r, avail, args, all_perturbed_onsets)
            print(f"  {r['name']:50s}  onset={r['onset']}  final={final}  eval_epochs={epochs_sel}")
        return

    ds_cache = {}
    for ds_name in datasets:
        try:
            ds, mask = _build_ood_dataset(ds_name, transform, data_root, download=download)
            ds_cache[ds_name] = (ds, mask)
            logger.info("[dataset] %s: %d images, %d classes", ds_name, len(ds), int(mask.sum()))
        except Exception as exc:
            logger.error("[dataset] %s failed: %s", ds_name, exc)

    if not ds_cache:
        logger.error("No datasets loaded — nothing to do.")
        return

    net, load_weights = build_timm_classifier(args.architecture, device)

    written = 0
    cached = 0

    for run_info in runs:
        run_name = run_info["name"]
        run_dir = run_info["dir"]
        avail = sorted(available_scratch_epochs(run_dir))

        epochs = _select_epochs(run_info, avail, args, all_perturbed_onsets)

        for epoch in epochs:
            if epoch not in set(avail):
                logger.warning("  [skip] %s epoch %d: no checkpoint", run_name, epoch)
                continue

            out_run_dir = output_dir / run_name
            out_run_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_run_dir / f"imagenet_ood_eval_epoch{epoch}_{preprocessing.tag}.json"

            existing_results = {}
            if out_file.exists() and not args.force:
                try:
                    existing = json.loads(out_file.read_text())
                    for r in existing.get("results", []):
                        existing_results[r["dataset"]] = r
                except Exception:
                    pass

            missing_ds = {k: v for k, v in ds_cache.items() if k not in existing_results}
            if not missing_ds:
                logger.info("  [skip] %s epoch %d (cached)", run_name, epoch)
                cached += 1
                continue

            ckpt_path = scratch_checkpoint_at_epoch(run_dir, epoch)
            logger.info("  [eval] %s epoch %d  ckpt=%s  (%d/%d datasets to run)",
                        run_name, epoch, ckpt_path.name, len(missing_ds), len(ds_cache))
            load_weights(ckpt_path)

            results = list(existing_results.values())
            for ds_name, (ds, mask) in missing_ds.items():
                t0 = time.time()
                top1, top5, n_samples = _classify_ood(
                    net, ds, mask, device, args.batch_size, args.num_workers, logger)
                elapsed = time.time() - t0

                block = {
                    "dataset": ds_name,
                    "epoch": epoch,
                    "architecture": args.architecture,
                    "run": run_name,
                    "preprocessing": preprocessing.tag,
                    "top1_accuracy": round(top1, 6),
                    "top5_accuracy": round(top5, 6) if top5 is not None else None,
                    "n_samples": n_samples,
                    "n_classes": int(mask.sum()),
                    "elapsed_s": round(elapsed, 1),
                }
                results.append(block)
                logger.info("    %s: top1=%.4f top5=%s  (%d samples, %.1fs)",
                            ds_name, top1,
                            f"{top5:.4f}" if top5 is not None else "n/a",
                            n_samples, elapsed)

            record = {
                "architecture": args.architecture,
                "run": run_name,
                "epoch": epoch,
                "preprocessing": preprocessing.tag,
                "results": results,
            }
            out_file.write_text(json.dumps(record, indent=2))
            written += 1

        del run_info
        gc.collect()

    logger.info("[sweep] done: %d written, %d cached -> %s", written, cached, output_dir)


if __name__ == "__main__":
    main()
