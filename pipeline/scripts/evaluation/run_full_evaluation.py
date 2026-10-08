"""Full-evaluation driver — runs every applicable eval suite for every
requested per-epoch checkpoint of a single trained run.

For each (epoch, eval_suite), invokes `pipeline/scripts/evaluation/run_inference.py`
as a subprocess so the parent process is isolated from per-eval CUDA / dataset
quirks. Writes one long-format `eval_results.csv` summarising every run.

Output schema (eval_results.csv):
    arch_name, training_mode, ckpt_epoch, eval_suite, dataset_or_roi, metric, value

Usage:
    python pipeline/scripts/evaluation/run_full_evaluation.py \\
        --sweep_config pipeline/configs/sweep_config.yaml \\
        --checkpoint_dir /path/to/run \\
        --arch_name resnet50_scratch \\
        --epochs 1,5,10,20,50,99
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent              # pipeline/scripts/evaluation
PIPELINE = EXP_DIR.parent.parent                       # pipeline/
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths  # noqa: E402

RUN_INFERENCE = EXP_DIR / "run_inference.py"

# Eval suite name -> the paths.yaml key holding its img_dir. Suites not listed
# (linear_probe) read their data without an img_dir override.
SUITE_IMG_DIR_KEY = {
    "behavioral_rsa":     "things_images",
    "brain_rsa_nod":      "nod_images",
    "brain_rsa_things":   "things_fmri_images",
    "brain_frrsa_nod":    "nod_images",
    "brain_frrsa_things": "things_fmri_images",
    "imagenet_val":       "imagenet_root",
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep_config", required=True, type=Path,
                   help="pipeline/configs/sweep_config.yaml")
    p.add_argument("--checkpoint_dir", required=True, type=Path,
                   help="Run directory containing per-epoch checkpoints.")
    p.add_argument("--arch_name", required=True, type=str,
                   help="Architecture name as it appears in sweep_config.architectures[].name")
    p.add_argument("--epochs", type=str, default=None,
                   help="Comma-separated epoch list. Defaults to sweep_config.eval_epochs.")
    p.add_argument("--eval_suites", type=str, default=None,
                   help="Comma-separated suite names to run. Defaults to all applicable.")
    p.add_argument("--save_dir", type=Path, default=None,
                   help="Output directory for eval JSONs and the aggregated CSV. "
                        "Defaults to <checkpoint_dir>/eval_results.")
    p.add_argument("--force", action="store_true",
                   help="Re-run evaluations even if their result JSONs exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the (epoch, suite) jobs that would run and exit.")
    return p.parse_args()


def _load_sweep_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _arch_entry(sweep_cfg, name):
    for entry in sweep_cfg["architectures"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Architecture '{name}' not in sweep_config.architectures")


def _applicable_suites(sweep_cfg, training_mode, requested=None):
    out = {}
    for name, spec in sweep_cfg["eval_suites"].items():
        if requested is not None and name not in requested:
            continue
        if training_mode in spec.get("applies_to", []):
            out[name] = spec
    return out


def _resolve_epochs(sweep_cfg, override):
    if override:
        return [int(e.strip()) for e in override.split(",") if e.strip()]
    return list(sweep_cfg.get("eval_epochs", []))


def _find_checkpoint(ckpt_dir: Path, training_mode: str, epoch: int) -> Path | None:
    """Locate the checkpoint file for `epoch` under `ckpt_dir`.

    Four checkpoint formats are supported:
      - Finetune (CLIP-HBA + DoRA)        -> epoch{E}_dora_params.pth
      - Scratch (timm)                    -> epoch{E}_model.pth
      - clip_contrastive (Lightning)      -> *epoch_epoch={E:03d}.ckpt
                                             (matches both `last.ckpt` siblings and
                                              `monitor_*-epoch_epoch={E:03d}.ckpt`)
      - scratch_lightning (Lightning)     -> epoch_{E:03d}.ckpt or
                                             epoch_epoch={E:03d}.ckpt under
                                             <ckpt_dir>/checkpoints/all_epochs/
    """
    if training_mode == "finetune":
        candidates = list(ckpt_dir.rglob(f"epoch{epoch}_dora_params.pth"))
    elif training_mode == "scratch":
        candidates = list(ckpt_dir.rglob(f"epoch{epoch}_model.pth"))
    elif training_mode == "clip_contrastive":
        # Lightning checkpoint names like `monitor_val_cifar100_epoch_acc1-epoch_epoch=017.ckpt` or
        # `epoch_epoch=017.ckpt`. Always 3-digit, zero-padded.
        epoch_str = f"{int(epoch):03d}"
        candidates = list(ckpt_dir.rglob(f"*epoch_epoch={epoch_str}.ckpt"))
        if not candidates:
            # Some runs use `epoch_NNN.ckpt` without the `epoch=` prefix.
            candidates = list(ckpt_dir.rglob(f"epoch_{epoch_str}.ckpt"))
    elif training_mode == "scratch_lightning":
        # runs_root Lightning checkpoints (RN50/ViT-B/16/ConvNeXt-B):
        # <run>/checkpoints/all_epochs/epoch_{E:03d}.ckpt (some resumed runs use
        # the epoch_epoch={E:03d}.ckpt variant instead).
        epoch_str = f"{int(epoch):03d}"
        candidates = list(ckpt_dir.rglob(f"epoch_{epoch_str}.ckpt"))
        if not candidates:
            candidates = list(ckpt_dir.rglob(f"epoch_epoch={epoch_str}.ckpt"))
    else:
        raise ValueError(
            f"Unknown training_mode {training_mode!r}; expected one of "
            "{'finetune', 'scratch', 'scratch_lightning', 'clip_contrastive'}."
        )
    canonical = [c for c in candidates if "_backup" not in str(c)]
    if not canonical:
        canonical = candidates
    if not canonical:
        return None
    if len(canonical) > 1:
        # For Lightning ckpts, multiple monitors can save at the same epoch
        # (e.g. monitor_val_cifar100 + monitor_val_cifar10). Prefer the first
        # alphabetically — they're functionally identical for our purposes.
        if training_mode == "clip_contrastive":
            return sorted(canonical)[0]
        raise RuntimeError(f"Ambiguous: multiple checkpoints for epoch {epoch}: {canonical}")
    return canonical[0]


def _run_one_eval(suite_name, suite_config, arch_entry, ckpt_path, epoch, save_dir, force):
    """Invoke run_inference.py for a single (suite, epoch). Returns the result JSON path."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(RUN_INFERENCE),
        "--config", str(suite_config),
        "--save_path", str(save_dir),
    ]
    if force:
        cmd.append("--force")

    # Push arch overrides through --set so we don't have to mutate the suite YAML.
    overrides = {
        "model_weights_path": str(ckpt_path),
        "epoch": int(epoch),
        "training_mode": arch_entry["training_mode"],
    }
    for k in ("architecture", "clip_hba_backbone", "vision_layers",
              "transformer_layers", "rank", "num_classes", "pretrained"):
        if k in arch_entry:
            overrides[k] = arch_entry[k]

    img_dir_key = SUITE_IMG_DIR_KEY.get(suite_name)
    if img_dir_key:
        overrides["img_dir"] = str(paths.path(img_dir_key))

    for k, v in overrides.items():
        cmd.extend(["--set", f"{k}={v}"])

    print(f"  [{suite_name} | epoch {epoch}] -> {' '.join(cmd[-6:])}", flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"  [WARN] {suite_name} epoch {epoch} exited with code {completed.returncode}",
              file=sys.stderr)

    return _collect_result_jsons(save_dir, suite_name, epoch)


def _collect_result_jsons(save_dir: Path, suite_name: str, epoch: int) -> list[Path]:
    """Return all JSONs that the eval just wrote (per-suite naming varies)."""
    suite_to_pattern = {
        "behavioral_rsa": f"inference_results_*_behavioral_epoch{epoch}.json",
        "brain_rsa_nod":  f"inference_results_nod_neural_epoch{epoch}.json",
        "brain_rsa_things": f"inference_results_things_neural_epoch{epoch}.json",
        "brain_frrsa_nod": f"inference_results_nod_frrsa*_epoch{epoch}.json",
        "brain_frrsa_things": f"inference_results_things_frrsa*_epoch{epoch}.json",
        "imagenet_val":   f"inference_results_imagenet_val_epoch{epoch}.json",
        "linear_probe":   f"inference_results_linear_probe_epoch{epoch}.json",
    }
    pattern = suite_to_pattern.get(suite_name, f"inference_results_*_epoch{epoch}.json")
    return sorted(save_dir.glob(pattern))


def _flatten_to_rows(arch_entry, suite_name, epoch, json_path):
    """Flatten a result JSON into long-format rows."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip-flatten] {json_path}: {exc}", file=sys.stderr)
        return []

    base = {
        "arch_name": arch_entry["name"],
        "training_mode": arch_entry["training_mode"],
        "ckpt_epoch": int(epoch),
        "eval_suite": suite_name,
    }

    rows = []

    if suite_name == "imagenet_val":
        for metric in ("top1_accuracy", "top5_accuracy"):
            if metric in data:
                rows.append({**base, "dataset_or_roi": "imagenet1k", "metric": metric, "value": float(data[metric])})

    elif suite_name in ("behavioral_rsa", "brain_rsa_nod", "brain_rsa_things"):
        # data is {roi_or_rdm_name: {... "score": rho, ...}, ...}
        for roi, block in data.items():
            if isinstance(block, dict) and "score" in block:
                rows.append({**base, "dataset_or_roi": roi, "metric": "rho", "value": float(block["score"])})
                if "p_value" in block:
                    rows.append({**base, "dataset_or_roi": roi, "metric": "p_value", "value": float(block["p_value"])})

    elif suite_name in ("brain_frrsa_nod", "brain_frrsa_things"):
        # data is {roi: {"cv_score", "insample_score", "cv_score_std", "alpha", "n_layers", ...}}
        for roi, block in data.items():
            if not isinstance(block, dict):
                continue
            for metric in ("cv_score", "insample_score", "cv_score_std", "alpha", "n_layers"):
                if block.get(metric) is not None:
                    rows.append({**base, "dataset_or_roi": roi, "metric": metric, "value": float(block[metric])})

    elif suite_name == "linear_probe":
        for ds_name, block in data.items():
            if isinstance(block, dict) and "score" in block and block["score"] is not None:
                rows.append({
                    **base, "dataset_or_roi": ds_name,
                    "metric": block.get("metric", "score"),
                    "value": float(block["score"]),
                })

    else:
        # Generic fallback: pull any numeric scalar.
        for k, v in (data or {}).items():
            if isinstance(v, (int, float)):
                rows.append({**base, "dataset_or_roi": "", "metric": k, "value": float(v)})

    return rows


def main():
    args = _parse_args()

    sweep_cfg = _load_sweep_config(args.sweep_config)
    arch = _arch_entry(sweep_cfg, args.arch_name)

    requested_suites = (
        {s.strip() for s in args.eval_suites.split(",") if s.strip()}
        if args.eval_suites else None
    )
    suites = _applicable_suites(sweep_cfg, arch["training_mode"], requested_suites)
    epochs = _resolve_epochs(sweep_cfg, args.epochs)

    if not suites:
        print(f"No applicable eval suites for arch '{args.arch_name}' "
              f"(training_mode={arch['training_mode']}). Exiting.", file=sys.stderr)
        sys.exit(2)
    if not epochs:
        print("No eval epochs specified (set sweep_config.eval_epochs or pass --epochs).",
              file=sys.stderr)
        sys.exit(2)

    save_root = args.save_dir or (args.checkpoint_dir / "eval_results")
    save_root.mkdir(parents=True, exist_ok=True)

    # Write the manifest so the aggregator knows what was attempted.
    manifest = {
        "arch_name": args.arch_name,
        "training_mode": arch["training_mode"],
        "checkpoint_dir": str(args.checkpoint_dir),
        "epochs": epochs,
        "eval_suites": list(suites.keys()),
    }
    with open(save_root / "eval_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if args.dry_run:
        print("[dry-run] Would run:")
        for ep in epochs:
            ckpt = _find_checkpoint(args.checkpoint_dir, arch["training_mode"], ep)
            for s in suites:
                print(f"  epoch={ep:>3}  suite={s:<18}  ckpt={ckpt}")
        return

    all_rows = []
    for epoch in epochs:
        ckpt = _find_checkpoint(args.checkpoint_dir, arch["training_mode"], epoch)
        if ckpt is None:
            print(f"  [skip-epoch] no checkpoint for epoch {epoch}", file=sys.stderr)
            continue

        for suite_name, spec in suites.items():
            suite_config = (REPO_ROOT / spec["config"]).resolve()
            json_paths = _run_one_eval(
                suite_name, suite_config, arch, ckpt, epoch, save_root, args.force
            )
            for jp in json_paths:
                all_rows.extend(_flatten_to_rows(arch, suite_name, epoch, jp))

    # Write the aggregated long-format CSV.
    csv_path = save_root / "eval_results.csv"
    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f"\nWrote {len(df)} rows -> {csv_path}")
    else:
        print(f"\nNo result rows produced. (See {save_root} for individual eval logs.)")


if __name__ == "__main__":
    main()
