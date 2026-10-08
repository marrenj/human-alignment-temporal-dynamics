"""SPoSE 1854-image RSA/CKA with BN-recalibrated ResNet-50 checkpoints.

ResNet-50's BatchNorm running statistics get corrupted by perturbation
training (image_noise, uniform_images). The standard behavioral RSA sweep
(run_behavioral_rsa.py) extracts embeddings in eval mode, so those corrupted statistics distort the
representations and the RSA/CKA scores. This script recalibrates BN buffers
on clean ImageNet train images before extracting THINGS embeddings, producing
corrected RSA/CKA numbers.

Writes results in the same aggregate-JSON format as run_behavioral_rsa.py, to
`<results_root>/posthoc_rdm1854/<run>/..._rsa_summary_bn_recal_in1k_bilinear.json`.

`--sub_epoch` scores the sub-epoch checkpoints of the nine
`*_recovery_dynamics` runs instead (`checkpoints/sub_epoch/`, the first epoch
after the perturbed one). Their records are keyed by step, the 0-based batch
index in the checkpoint filename, exactly as `run_behavioral_rsa.py --sub_epoch`
writes them.

Usage:
    python pipeline/scripts/evaluation/run_behavioral_rsa_bn_recalibrated.py
    python pipeline/scripts/evaluation/run_behavioral_rsa_bn_recalibrated.py --device cuda:1
    python pipeline/scripts/evaluation/run_behavioral_rsa_bn_recalibrated.py --sub_epoch --dry-run
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE = EVAL_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths
from temporal_perturbations.utils.logging import setup_logger
from temporal_perturbations.models.lightning_ckpt import load_lightning_state_dict
from temporal_perturbations.models.scratch.scratch_utils import (
    build_cross_arch_model,
    embed_batched,
    sorted_subepoch_checkpoints,
    subepoch_parent_epoch,
)
from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset
from temporal_perturbations.data.eval_transforms import resolve_preprocessing
from temporal_perturbations.inference.inference_core import (
    compute_model_rdm,
    compute_rdm_similarity,
    build_spose_target_rdm,
    compute_cka,
)
sys.path.insert(0, str(EVAL_DIR))
from run_behavioral_rsa import _build_spose_matrix  # noqa: E402

DEFAULT_RESULTS = paths.results_root() / 'posthoc_rdm1854'
DEFAULT_CONFIG = PIPELINE / 'configs' / 'inference_config_posthoc_1854.yaml'

ONSET_EPOCH = {'early': 5, 'mid': 30, 'late': 70}
PERTS = ['label_shuffle', 'image_noise', 'uniform_images']
TIMINGS = ['early', 'mid', 'late']

IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CALIB_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.PILToTensor(),
])


def _run_name(pert, timing, seed_suffix=''):
    return f'resnet50_imagenet_{pert}_{timing}{seed_suffix}'


def _ckpt_path(runs_root, run, epoch):
    base = runs_root / run / 'checkpoints' / 'all_epochs'
    for name in (f'epoch_{epoch:03d}.ckpt', f'epoch_epoch={epoch:03d}.ckpt'):
        if (base / name).is_file():
            return base / name
    return None


def _discover_epochs(runs_root, run):
    """Return sorted list of all available checkpoint epochs for a run."""
    base = runs_root / run / 'checkpoints' / 'all_epochs'
    if not base.is_dir():
        return []
    epochs = set()
    for f in base.iterdir():
        name = f.name
        if name.startswith('epoch_') and name.endswith('.ckpt'):
            try:
                part = name.replace('epoch_epoch=', '').replace('epoch_', '').replace('.ckpt', '')
                epochs.add(int(part))
            except ValueError:
                continue
    return sorted(epochs)


def _build_calib_set(imagenet_train, n_images, logger):
    """Load n_images clean ImageNet train images as a uint8 tensor."""
    per_class = max(1, n_images // 1000)
    xs = []
    class_dirs = sorted(p for p in imagenet_train.iterdir() if p.is_dir())
    for idx, cdir in enumerate(class_dirs):
        for f in sorted(cdir.iterdir())[:per_class]:
            try:
                xs.append(CALIB_TF(Image.open(f).convert('RGB')))
            except OSError:
                continue
        if len(xs) >= n_images:
            break
    x = torch.stack(xs[:n_images])
    logger.info("  BN calibration set: %s (%d images)", tuple(x.shape), len(x))
    return x


def _norm_batch(batch_u8, device):
    x = batch_u8.to(device, non_blocking=True).float().div_(255)
    return (x - IN_MEAN.to(device)) / IN_STD.to(device)


@torch.no_grad()
def _recalibrate_bn(model, calib_x, device, batch_size, logger):
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        logger.info("  No BatchNorm layers found — skipping recalibration.")
        return 0
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for i in range(0, len(calib_x), batch_size):
        model(_norm_batch(calib_x[i:i + batch_size], device))
    for m in bns:
        m.eval()
    logger.info("  Recalibrated %d BatchNorm layers on %d clean images.",
                len(bns), len(calib_x))
    return len(bns)


def _preload_things_images(config, preprocessing, logger):
    ds = ThingsBehavioralDataset(
        img_annotations_file=config["annotations_file"],
        img_dir=config["img_dir"],
        transform=preprocessing.build(224),
    )
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config.get("num_workers", 4), pin_memory=False)
    names, chunks = [], []
    for image_names, images, _ in loader:
        names.extend(list(image_names))
        chunks.append(images)
    images = torch.cat(chunks, dim=0)
    logger.info("  Preloaded %d THINGS images -> %s", len(names), tuple(images.shape))
    return images, names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--calib_images', type=int, default=8192)
    ap.add_argument('--n_permutations', type=int, default=10000)
    ap.add_argument('--output_dir', type=Path, default=DEFAULT_RESULTS)
    ap.add_argument('--runs_root', type=Path, default=None, help='Default: paths key runs_root.')
    ap.add_argument('--img_dir', type=Path, default=None,
                    help="Override the config's img_dir (the THINGS image root).")
    ap.add_argument('--imagenet_train', type=Path, default=None,
                    help='Clean BN calibration images. Default: <imagenet_root>/train.')
    ap.add_argument('--epochs', default=None,
                    help='Comma-separated epochs, or "all" to discover every checkpoint. '
                         'Default: onset epochs per timing.')
    ap.add_argument('--runs', default='onset',
                    choices=['onset', 'all'],
                    help='"onset" (default) evaluates each run at its onset epoch; '
                         '"all" evaluates baseline + all 9 perturbed runs at --epochs.')
    ap.add_argument('--seed_suffix', default='',
                    help='Appended to run dir names (e.g. "_seed43") to target '
                         'a different seed family. Empty string = original seed42 runs.')
    ap.add_argument('--sub_epoch', action='store_true',
                    help='Score the sub-epoch checkpoints of the '
                         '*_recovery_dynamics runs (see the module docstring).')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.sub_epoch and (args.runs == 'all' or args.epochs is not None):
        ap.error('--sub_epoch scores every sub-epoch checkpoint; drop --runs/--epochs')

    from temporal_perturbations.utils.load_yaml_config import load_yaml_config
    config = load_yaml_config(DEFAULT_CONFIG)
    if args.img_dir is not None:
        config["img_dir"] = str(args.img_dir)
    p = Path(config["annotations_file"])
    if not p.is_absolute():
        config["annotations_file"] = str(REPO_ROOT / p)
    runs_root = args.runs_root or paths.path("runs_root")
    imagenet_train = args.imagenet_train or paths.path("imagenet_root") / "train"

    SUFFIX = '_bn_recal_in1k_bilinear'
    dataset = config["dataset"]
    eval_type = config["evaluation_type"]
    rsa_similarity_metric = config["rsa_similarity_metric"]
    model_rdm_distance_metric = config["model_rdm_distance_metric"]
    reference_rdm_distance_metric = config["reference_rdm_distance_metric"]
    cka_unbiased = bool(config.get("cka_unbiased", False))

    ssuf = args.seed_suffix
    # (run, perturbation, timing, record key, checkpoint or None, extra record fields)
    jobs = []
    if args.sub_epoch:
        for t in TIMINGS:
            for pert in PERTS:
                run = f'{_run_name(pert, t, ssuf)}_recovery_dynamics'
                sub_dir = runs_root / run / 'checkpoints' / 'sub_epoch'
                ckpts = sorted_subepoch_checkpoints(sub_dir)
                extra = {'sub_epoch': True, 'parent_epoch': subepoch_parent_epoch(sub_dir),
                         'total_steps': ckpts[-1][0]}
                for step, ckpt in ckpts:
                    jobs.append((run, pert, t, step, ckpt, dict(extra, step=step)))
    elif args.runs == 'all':
        if args.epochs is None:
            ap.error('--runs all requires --epochs')
        all_runs = [f'resnet50_imagenet_baseline{ssuf}']
        for pert in PERTS:
            for t in TIMINGS:
                all_runs.append(_run_name(pert, t, ssuf))
        for run in all_runs:
            if args.epochs == 'all':
                epochs = _discover_epochs(runs_root, run)
            else:
                epochs = [int(e) for e in args.epochs.split(',')]
            for ep in epochs:
                jobs.append((run, 'none', 'all', ep, _ckpt_path(runs_root, run, ep), {}))
    else:
        for t in TIMINGS:
            ep = ONSET_EPOCH[t]
            base = f'resnet50_imagenet_baseline{ssuf}'
            jobs.append((base, 'none', t, ep, _ckpt_path(runs_root, base, ep), {}))
            for pert in PERTS:
                run = _run_name(pert, t, ssuf)
                jobs.append((run, pert, t, ep, _ckpt_path(runs_root, run, ep), {}))

    def _epoch_cached(agg_path, epoch):
        if not agg_path.is_file():
            return False
        try:
            with agg_path.open() as f:
                agg = json.load(f)
            return any(str(r.get('epoch')) == str(epoch) for r in agg.get('results', []))
        except (OSError, ValueError):
            return False

    print(f'{len(jobs)} checkpoints to evaluate')
    for run, pert, t, ep, _, _ in jobs:
        out_dir = args.output_dir / run
        agg_json = out_dir / f'test_things_behavioral_full1854_rsa_summary{SUFFIX}.json'
        status = '[skip]' if _epoch_cached(agg_json, ep) and not args.force else '[todo]'
        print(f'  {status} {run} {"step" if args.sub_epoch else "ep"}{ep}')
    if args.dry_run:
        return

    device = torch.device(args.device)
    log_path = args.output_dir / 'bn_recal_spose_sweep_log.txt'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(log_path))

    logger.info("Building calibration set...")
    calib_x = _build_calib_set(imagenet_train, args.calib_images, logger)

    preprocessing = resolve_preprocessing('in1k_bilinear')
    logger.info("Loading THINGS images with %s preprocessing...", 'in1k_bilinear')
    images_cpu, image_names = _preload_things_images(config, preprocessing, logger)

    logger.info("Building SPoSE reference RDM and matrix...")
    reference_vec = build_spose_target_rdm(
        config["annotations_file"], reference_rdm_distance_metric,
        image_order=image_names)
    n_images = len(image_names)
    reference_rdm_full = np.zeros((n_images, n_images), dtype=np.float64)
    reference_rdm_full[np.triu_indices(n_images, k=1)] = reference_vec
    reference_rdm_full += reference_rdm_full.T
    spose_matrix = _build_spose_matrix(config["annotations_file"], image_names)

    logger.info("Building ResNet-50 model...")
    model, load_weights = build_cross_arch_model('RN50', device)

    for run, pert, timing, epoch, ckpt, extra in jobs:
        out_dir = args.output_dir / run
        out_dir.mkdir(parents=True, exist_ok=True)
        agg_json = out_dir / f'test_things_behavioral_full1854_rsa_summary{SUFFIX}.json'

        if _epoch_cached(agg_json, epoch) and not args.force:
            logger.info("[skip] %s ep%d — already exists", run, epoch)
            continue

        if ckpt is None:
            logger.warning("[MISSING] no checkpoint for %s epoch %d", run, epoch)
            continue

        logger.info("\n=== %s %s %d ===", run, "step" if args.sub_epoch else "epoch", epoch)
        model = load_weights(str(ckpt))
        n_bn = _recalibrate_bn(model, calib_x, device, args.batch_size, logger)

        embeddings_t = torch.from_numpy(
            embed_batched(model, images_cpu, config["batch_size"], device))
        logger.info("  Embeddings shape: %s", tuple(embeddings_t.shape))

        model_rdm = compute_model_rdm(
            embeddings_t, dataset_name=dataset,
            annotations_file=config["annotations_file"],
            categories=None, distance_metric=model_rdm_distance_metric,
        )

        ref_vec_epoch = reference_vec
        n_dropped = 0
        if np.isnan(model_rdm).any():
            n = model_rdm.shape[0]
            off_diag_nan = np.array([
                np.isnan(model_rdm[i, np.arange(n) != i]).all() for i in range(n)
            ])
            nan_rows = np.where(off_diag_nan)[0]
            if len(nan_rows) > 0:
                logger.warning("  Dropping %d degenerate images from RDM", len(nan_rows))
                keep = np.ones(n, dtype=bool)
                keep[nan_rows] = False
                model_rdm = model_rdm[np.ix_(keep, keep)]
                ref_sub = reference_rdm_full[np.ix_(keep, keep)]
                ref_vec_epoch = ref_sub[np.triu_indices_from(ref_sub, k=1)]
                n_dropped = len(nan_rows)

        model_rdm_vec = model_rdm[np.triu_indices_from(model_rdm, k=1)]
        rho, p_value = compute_rdm_similarity(
            model_rdm_vec, ref_vec_epoch,
            similarity_metric=rsa_similarity_metric,
            n_permutations=args.n_permutations,
        )
        rsa_block = {
            "epoch": str(epoch),
            "evaluation_type": eval_type,
            "dataset": dataset,
            "reference_rdm_name": "spose_1854",
            "score": float(rho),
            "p_value": float(p_value),
            "p_method": "permutation",
            "n_permutations": args.n_permutations,
            "rsa_similarity_metric": rsa_similarity_metric,
            "model_rdm_distance_metric": model_rdm_distance_metric,
            "reference_rdm_distance_metric": reference_rdm_distance_metric,
            "bn_recalibrated": True,
            "n_bn_layers": n_bn,
            "calib_images": args.calib_images,
        }
        if n_dropped > 0:
            rsa_block["n_degenerate_images_dropped"] = n_dropped
        logger.info("  RSA rho=%.4f  p=%.4g", rho, p_value)

        emb_np = embeddings_t.detach().cpu().numpy().astype(np.float64)
        cka = compute_cka(emb_np, spose_matrix, unbiased=cka_unbiased)
        logger.info("  CKA=%.4f", cka)

        record = {
            "epoch": str(epoch),
            "run_name": run,
            "dataset": dataset,
            "evaluation_type": eval_type,
            "architecture": "resnet50_imagenet",
            "bn_recalibrated": True,
            **extra,
            "rsa_results": {"behavioral_rdm": rsa_block},
            "cka_results": {
                "behavioral_cka": {
                    "epoch": str(epoch),
                    "evaluation_type": eval_type,
                    "dataset": dataset,
                    "reference_name": "spose_1854",
                    "score": float(cka),
                    "kernel": "linear",
                    "unbiased": cka_unbiased,
                    "bn_recalibrated": True,
                }
            },
        }

        if agg_json.is_file():
            with agg_json.open() as f:
                agg = json.load(f)
        else:
            agg = {"results": []}
        if extra:
            agg.update({k: extra[k] for k in ('sub_epoch', 'parent_epoch', 'total_steps')})
        agg["results"] = [
            r for r in agg.get("results", []) if str(r.get("epoch")) != str(epoch)
        ]
        agg["results"].append(record)
        with agg_json.open('w') as f:
            json.dump(agg, f, indent=2)
        logger.info("  Wrote %s", agg_json)

        del embeddings_t, model_rdm, emb_np
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
