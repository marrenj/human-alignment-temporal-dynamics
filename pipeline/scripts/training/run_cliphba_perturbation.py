"""Run one CLIP-HBA perturbation training task; the sweep grid (sweep_config.yaml) lives in the caller."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXP_DIR    = Path(__file__).resolve().parent          # pipeline/scripts/training
PIPELINE   = EXP_DIR.parent.parent                    # pipeline/
REPO_ROOT  = PIPELINE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from train_cliphba import _prepare_single_run, TRAINING_INT_KEYS, TRAINING_FLOAT_ONLY
from temporal_perturbations.training.trainer import run_training_experiment
from temporal_perturbations.utils.load_yaml_config import load_yaml_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single (perturb_type, perturb_epoch, perturb_length, perturb_seed) "
            "training task. The grid is defined by the caller (e.g. the pipeline notebook)."
        )
    )
    parser.add_argument("--config", type=Path, default=PIPELINE / "configs" / "training_config.yaml",
                        help="Base YAML config.")
    parser.add_argument("--perturb_type", required=True,
                        choices=["random_target", "label_shuffle", "image_noise", "uniform_images"],
                        help="Which perturbation strategy to apply.")
    parser.add_argument("--perturb_epoch", type=int, required=True,
                        help="Start epoch (0-indexed) at which the perturbation begins.")
    parser.add_argument("--perturb_length", type=int, default=1,
                        help="Number of consecutive epochs to apply the perturbation.")
    parser.add_argument("--perturb_seed", type=int, default=1,
                        help="Random seed for the perturbation strategy.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Override random_seed (model init) from the config.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Override save_path in config.")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Override img_dir in config.")
    parser.add_argument("--baseline_checkpoint_path", type=str, default=None,
                        help="Path to a completed baseline run directory. Required when perturb_epoch > 0.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional run tag suffix.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Bypass idempotency cache (passthrough to trainer).")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                        help="Print the resolved task config and exit.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = args.config.resolve()
    config = load_yaml_config(
        config_path,
        numeric_keys=TRAINING_INT_KEYS,
        float_only=TRAINING_FLOAT_ONLY,
    )

    if args.save_path:
        config["save_path"] = args.save_path
    if args.img_dir:
        config["img_dir"] = args.img_dir
    if args.random_seed is not None:
        config["random_seed"] = args.random_seed

    config["perturb_type"]   = args.perturb_type
    config["perturb_epoch"]  = args.perturb_epoch
    config["perturb_length"] = args.perturb_length
    config["perturb_seed"]   = args.perturb_seed

    if args.perturb_epoch > 0:
        if not args.baseline_checkpoint_path:
            raise SystemExit(
                f"--baseline_checkpoint_path is required for perturb_epoch={args.perturb_epoch} > 0"
            )
        config["baseline_checkpoint_path"] = args.baseline_checkpoint_path

    if args.tag:
        config["run_tag"] = args.tag

    prepared_config = _prepare_single_run(config, config_path)

    if args.dry_run:
        print("--- DRY RUN: resolved task config ---")
        for key, value in prepared_config.items():
            print(f"  {key}: {value}")
        return

    run_training_experiment(prepared_config)


if __name__ == "__main__":
    main()
