"""Extract per-epoch ImageNet val curves from the runs_root TensorBoard logs.

The from-scratch runs (Lightning, temporal_perturbations/scratch_training) log validation metrics
every epoch to `<run>/version_*/events.out.tfevents.*`. This script stitches those
scalars into one JSON keyed by run name, so the plotting code never has to touch
tfevents.

Resumed runs write a fresh `version_N` per resume, and the resume epoch appears in
both the old and the new version dir. Versions are merged in numeric order with
later versions winning, which keeps the post-resume value at the seam.

`clip_vitb32_*` is contrastive (InfoNCE) and logs no ImageNet validation scalars;
those runs are reported as skipped.

Usage
-----
    python pipeline/scripts/evaluation/extract_imagenet_val_curves.py
    python pipeline/scripts/evaluation/extract_imagenet_val_curves.py --force
"""

from pathlib import Path
import argparse
import json
import re
import sys

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))
from temporal_perturbations import paths  # noqa: E402


# TensorBoard scalar tag -> key in the emitted JSON.
TAGS = {
    "val/imagenet/acc1":     "val_acc1",
    "val/imagenet/acc5":     "val_acc5",
    "val/imagenet/ce_loss":  "val_ce_loss",
    "train/acc1":            "train_acc1",
    "train/ce_loss_epoch":   "train_ce_loss",
}

_VERSION_RE = re.compile(r"version_(\d+)$")


def _version_dirs(run_dir: Path) -> list[Path]:
    dirs = [d for d in run_dir.glob("version_*") if _VERSION_RE.search(d.name)]
    return sorted(dirs, key=lambda d: int(_VERSION_RE.search(d.name).group(1)))


def _scalars_by_epoch(version_dir: Path) -> dict[str, dict[int, float]]:
    """Return {json_key: {epoch: value}} for one version dir.

    Lightning logs against global step, so the `epoch` scalar is used to map each
    validation point back to its epoch (steps differ per architecture).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])
    if "epoch" not in available:
        return {}

    step_to_epoch = {s.step: int(s.value) for s in ea.Scalars("epoch")}

    def epoch_for(step):
        if step in step_to_epoch:
            return step_to_epoch[step]
        return step_to_epoch[min(step_to_epoch, key=lambda k: abs(k - step))]

    out = {}
    for tag, key in TAGS.items():
        if tag not in available:
            continue
        out[key] = {epoch_for(s.step): float(s.value) for s in ea.Scalars(tag)}
    return out


def extract_run(run_dir: Path) -> dict:
    """Merge every version dir of one run into {json_key: {epoch: value}}."""
    merged: dict[str, dict[int, float]] = {}
    for vdir in _version_dirs(run_dir):
        for key, series in _scalars_by_epoch(vdir).items():
            merged.setdefault(key, {}).update(series)   # later version wins at the seam
    return merged


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", type=Path, default=None, help="Default: paths key runs_root.")
    p.add_argument("--output", type=Path,
                   default=paths.results_root() / "imagenet_val_curves.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = p.parse_args()
    args.runs_root = args.runs_root or paths.path("runs_root")

    if args.output.exists() and not args.force:
        print(f"[curves] {args.output} exists; pass --force to rebuild.")
        return

    run_dirs = sorted(d for d in args.runs_root.iterdir() if d.is_dir())
    if args.dry_run:
        for d in run_dirs:
            print(f"  {d.name}  ({len(_version_dirs(d))} version dirs)")
        print(f"--- would write: {args.output} ---")
        return

    curves, skipped = {}, []
    for d in run_dirs:
        series = extract_run(d)
        if not series.get("val_acc1"):
            skipped.append(d.name)
            continue
        epochs = sorted(series["val_acc1"])
        curves[d.name] = {
            key: {str(e): v for e, v in sorted(vals.items())}
            for key, vals in series.items()
        }
        gaps = [e for e in range(epochs[0], epochs[-1] + 1) if e not in series["val_acc1"]]
        print(f"[curves] {d.name:48s} epochs {epochs[0]:3d}..{epochs[-1]:3d} "
              f"(n={len(epochs)})" + (f"  GAPS: {gaps}" if gaps else ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "source": str(args.runs_root),
            "description": "Per-epoch ImageNet validation metrics stitched across "
                           "version_* TensorBoard logs; later versions win at resume seams. "
                           "Accuracies are fractions, not percentages.",
            "runs": curves,
        }, f, indent=2)

    print(f"\n[curves] wrote {len(curves)} runs -> {args.output}")
    if skipped:
        print(f"[curves] skipped {len(skipped)} run(s) with no val/imagenet/acc1 "
              f"(contrastive runs log no ImageNet validation): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
