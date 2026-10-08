"""Extract per-epoch learning rate from the runs_root TensorBoard logs.

Reads the ``lr-SGD/pg1`` scalar from each baseline run's TensorBoard events,
maps global_step to epoch via the ``epoch`` scalar, and takes the last step's
LR value per epoch (the LR that applied for the majority of that epoch).

Only baseline runs are extracted — perturbed runs fork from the same schedule.

Resumed runs write a fresh ``version_N`` per resume; versions are merged in
numeric order with later versions winning, matching extract_imagenet_val_curves.

Usage
-----
    python pipeline/scripts/evaluation/extract_lr_schedule.py
    python pipeline/scripts/evaluation/extract_lr_schedule.py --force
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


LR_TAG_PRIORITY = [
    "lr-SGD/pg1",
    "lr-AdamW/pg1",
    "lr-AdamW/params_with_wd",
]

BASELINES = [
    "resnet50_imagenet_baseline",
    "vitb16_imagenet_baseline",
    "convnext_b_imagenet_baseline",
    "clip_vitb32_baseline",
]

_VERSION_RE = re.compile(r"version_(\d+)$")


def _version_dirs(run_dir: Path) -> list[Path]:
    dirs = [d for d in run_dir.glob("version_*") if _VERSION_RE.search(d.name)]
    return sorted(dirs, key=lambda d: int(_VERSION_RE.search(d.name).group(1)))


def _find_lr_tag(available_tags: set[str]) -> str | None:
    for tag in LR_TAG_PRIORITY:
        if tag in available_tags:
            return tag
    return None


def _lr_by_epoch(version_dir: Path) -> dict[int, float]:
    """Return {epoch: lr} for one version dir, taking the last step per epoch."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])
    if "epoch" not in available:
        return {}
    lr_tag = _find_lr_tag(available)
    if lr_tag is None:
        return {}

    step_to_epoch = {s.step: int(s.value) for s in ea.Scalars("epoch")}

    def epoch_for(step):
        if step in step_to_epoch:
            return step_to_epoch[step]
        return step_to_epoch[min(step_to_epoch, key=lambda k: abs(k - step))]

    epoch_lr_pairs: dict[int, tuple[int, float]] = {}
    for s in ea.Scalars(lr_tag):
        ep = epoch_for(s.step)
        prev = epoch_lr_pairs.get(ep)
        if prev is None or s.step > prev[0]:
            epoch_lr_pairs[ep] = (s.step, float(s.value))

    return {ep: lr for ep, (_, lr) in epoch_lr_pairs.items()}


def extract_run(run_dir: Path) -> dict[int, float]:
    """Merge every version dir of one run into {epoch: lr}."""
    merged: dict[int, float] = {}
    for vdir in _version_dirs(run_dir):
        merged.update(_lr_by_epoch(vdir))
    return merged


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", type=Path, default=None, help="Default: paths key runs_root.")
    p.add_argument("--output", type=Path,
                   default=paths.results_root() / "lr_schedule.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = p.parse_args()
    args.runs_root = args.runs_root or paths.path("runs_root")

    if args.output.exists() and not args.force:
        print(f"[lr] {args.output} exists; pass --force to rebuild.")
        return

    if args.dry_run:
        for name in BASELINES:
            d = args.runs_root / name
            exists = d.is_dir()
            n_ver = len(_version_dirs(d)) if exists else 0
            print(f"  {name:48s} {'exists' if exists else 'MISSING'}  "
                  f"({n_ver} version dirs)")
        print(f"--- would write: {args.output} ---")
        return

    schedules = {}
    for name in BASELINES:
        d = args.runs_root / name
        if not d.is_dir():
            print(f"[lr] SKIP {name} (not found)")
            continue
        lr_map = extract_run(d)
        if not lr_map:
            print(f"[lr] SKIP {name} (no LR scalar in {LR_TAG_PRIORITY})")
            continue
        epochs = sorted(lr_map)
        schedules[name] = {str(e): lr_map[e] for e in epochs}
        print(f"[lr] {name:48s} epochs {epochs[0]:3d}..{epochs[-1]:3d} "
              f"(n={len(epochs)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "source": str(args.runs_root),
            "description": "Per-epoch learning rate (last step value per epoch) "
                           "extracted from lr-SGD/pg1 TensorBoard scalar. "
                           "Baseline runs only.",
            "runs": schedules,
        }, f, indent=2)

    print(f"\n[lr] wrote {len(schedules)} runs -> {args.output}")


if __name__ == "__main__":
    main()
