"""BN-recalibrated frozen linear-probe benchmark for ResNet-50.

ResNet-50's BatchNorm running statistics get corrupted by perturbation training
(image_noise, uniform_images). The standard linear probe eval extracts features
in eval mode with those corrupted statistics, distorting the representations.
This script recalibrates BN buffers on clean ImageNet train images before
feature extraction, producing corrected probe scores.

Wraps the existing probe_checkpoint infrastructure from run_linear_probe_eval.py
-- the only addition is a BN recalibration step between weight loading and
feature extraction. Results go to
`<results_root>/linear_probe_eval_scratch_bn_recal_std/`.

Usage:
    python pipeline/scripts/evaluation/run_linear_probe_eval_bn_recalibrated.py
    python pipeline/scripts/evaluation/run_linear_probe_eval_bn_recalibrated.py --dry-run
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR))

from run_linear_probe_eval import (  # noqa: E402
    discover_runs,
    probe_checkpoint,
    _is_complete,
)
from run_model_to_model_rsa import (  # noqa: E402
    build_model as build_scratch_model,
)
from temporal_perturbations import paths  # noqa: E402
from temporal_perturbations.data.eval_transforms import eval_preprocessing  # noqa: E402
from temporal_perturbations.inference.linear_probe_eval import DEFAULT_PROBE_SOLVER  # noqa: E402
from temporal_perturbations.models.scratch.scratch_utils import (  # noqa: E402
    scratch_checkpoint_at_epoch,
)

ARCHITECTURE = "RN50"
DEFAULT_OUTPUT_DIR = paths.results_root() / "linear_probe_eval_scratch_bn_recal_std"
DEFAULT_BN_FEATURE_CACHE = paths.path("feature_cache") / "probe_features_bn_recal"

IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CALIB_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.PILToTensor(),
])


def _build_calib_set(imagenet_train, n_images, logger):
    per_class = max(1, n_images // 1000)
    xs = []
    class_dirs = sorted(p for p in imagenet_train.iterdir() if p.is_dir())
    for cdir in class_dirs:
        for f in sorted(cdir.iterdir())[:per_class]:
            try:
                xs.append(CALIB_TF(Image.open(f).convert("RGB")))
            except OSError:
                continue
        if len(xs) >= n_images:
            break
    x = torch.stack(xs[:n_images])
    logger.info("BN calibration set: %s (%d images)", tuple(x.shape), len(x))
    return x


def _norm_batch(batch_u8, device):
    x = batch_u8.to(device, non_blocking=True).float().div_(255)
    return (x - IN_MEAN.to(device)) / IN_STD.to(device)


@torch.no_grad()
def _recalibrate_bn(model, calib_x, device, batch_size, logger):
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        logger.info("  No BatchNorm layers found.")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs_root", default=None, help="Default: paths key runs_root.")
    ap.add_argument("--imagenet_train", type=Path, default=None,
                    help="BN calibration images; default <imagenet_root>/train.")
    ap.add_argument("--runs", default="all", choices=["all", "baseline", "perturbed"])
    ap.add_argument("--perturb_type", default=None,
                    choices=["label_shuffle", "image_noise", "uniform_images"])
    ap.add_argument("--phase", default=None, choices=["early", "mid", "late"])
    ap.add_argument("--epochs", default=None,
                    help="Comma-separated epochs, or 'onset'. Default: each run's final epoch.")
    ap.add_argument("--datasets", default=(
        "vtab/cifar100,vtab/cifar10,vtab/caltech101,"
        "vtab/clevr_closest_object_distance,vtab/clevr_count_all,"
        "vtab/kitti_closest_vehicle_distance,vtab/dtd,vtab/eurosat,"
        "vtab/flowers,vtab/pets,vtab/resisc45,vtab/svhn,vtab/cars,"
        "vtab/food101,fgvc_aircraft,voc2007,country211,gtsrb,mnist,"
        "renderedsst2,stl10"))
    ap.add_argument("--data_root", default=None, help="Default: paths key probe_datasets.")
    ap.add_argument("--feature_cache", default=str(DEFAULT_BN_FEATURE_CACHE))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--calib_images", type=int, default=8192)
    ap.add_argument("--calib_batch_size", type=int, default=128)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--probe_solver", default=DEFAULT_PROBE_SOLVER)
    ap.add_argument("--probe_max_iter", type=int, default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="Explicit checkpoint path; bypasses runs_root discovery. "
                         "Requires --epoch and --run_name.")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Epoch label for --checkpoint.")
    ap.add_argument("--run_name", default=None,
                    help="Output subdir name for --checkpoint.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()
    args.runs_root = args.runs_root or paths.path("runs_root")
    args.data_root = args.data_root or str(paths.path("probe_datasets"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger("bn_recal_probe")
    device = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.checkpoint:
        if args.epoch is None or not args.run_name:
            raise SystemExit("--checkpoint requires --epoch and --run_name.")
        entries = [{"name": args.run_name, "type": None, "phase": None,
                    "is_baseline": False, "checkpoint": Path(args.checkpoint),
                    "epochs": [args.epoch]}]
    else:
        epochs_arg = ("onset" if args.epochs == "onset"
                      else {int(e) for e in args.epochs.split(",")} if args.epochs else None)
        entries = discover_runs(args.runs_root, ARCHITECTURE, args.runs,
                                args.perturb_type, args.phase, epochs_arg, logger)
    logger.info("[sweep] RN50 BN-recalibrated, %d runs", len(entries))

    if args.dry_run:
        for e in entries:
            print(f"{e['name']}  baseline={e['is_baseline']}  epochs={e['epochs']}")
        return

    out_root = Path(args.output_dir)
    preprocessing = eval_preprocessing(ARCHITECTURE)
    logger.info("[sweep] preprocessing=%s", preprocessing.tag)
    logger.info("[sweep] output root: %s", out_root)

    logger.info("Building calibration set (%d images)...", args.calib_images)
    calib_x = _build_calib_set(args.imagenet_train or paths.path("imagenet_root") / "train",
                               args.calib_images, logger)

    model, load_weights_raw = build_scratch_model(ARCHITECTURE, device)

    def load_weights_bn_recal(ckpt_path):
        m = load_weights_raw(ckpt_path)
        _recalibrate_bn(m, calib_x, device, args.calib_batch_size, logger)
        return m

    n_written = n_skipped = 0
    for e in entries:
        out_dir = out_root / e["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in e["epochs"]:
            out_json = out_dir / f"linear_probe_eval_epoch{epoch}.json"
            if not args.force and _is_complete(out_json, datasets, args.probe_solver,
                                               preprocessing.tag):
                logger.info("[skip] %s epoch%d (cached)", e["name"], epoch)
                n_skipped += 1
                continue

            ckpt = e.get("checkpoint") or scratch_checkpoint_at_epoch(e["run_dir"], epoch)
            blocks = probe_checkpoint(
                load_weights_bn_recal, ckpt, label=e["name"],
                probe_dir=out_dir / "probe",
                epoch=epoch, datasets=datasets, preprocessing=preprocessing,
                image_size=args.image_size, batch_size=args.batch_size,
                num_workers=args.num_workers, device=device, force=args.force,
                data_root=args.data_root, download=not args.no_download,
                probe_solver=args.probe_solver, probe_max_iter=args.probe_max_iter,
                feature_cache_dir=(Path(args.feature_cache) / e["name"]
                                   / f"epoch{epoch}"),
                logger=logger,
            )

            prior = {}
            if out_json.exists() and not args.force:
                try:
                    cached = json.loads(out_json.read_text())
                    if (cached.get("solver") == args.probe_solver
                            and cached.get("preprocessing") == preprocessing.tag):
                        prior = cached.get("results", {})
                except (OSError, ValueError):
                    pass
            merged = {**prior, **blocks}

            result = {
                "architecture": ARCHITECTURE,
                "run_name": e["name"],
                "is_baseline": e["is_baseline"],
                "perturb_type": e["type"], "phase": e["phase"],
                "epoch": epoch,
                "checkpoint": Path(ckpt).name,
                "bn_recalibrated": True,
                "calib_images": args.calib_images,
                "datasets": sorted(merged),
                "datasets_this_run": datasets,
                "solver": args.probe_solver,
                "preprocessing": preprocessing.tag,
                "results": merged,
            }
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            n_written += 1
            for ds, blk in blocks.items():
                top5 = blk.get("top5_accuracy")
                logger.info("[done] %s epoch%d %s: top1=%.4f top5=%s%s",
                            e["name"], epoch, ds, blk["score"],
                            "n/a" if top5 is None else f"{top5:.4f}",
                            "" if blk.get("converged", True) else "  [not-converged]")

        gc.collect()

    logger.info("[sweep] done: %d written, %d cached -> %s", n_written, n_skipped, out_root)


if __name__ == "__main__":
    main()
