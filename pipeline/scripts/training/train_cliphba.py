"""Train one CLIP-HBA (DoRA finetune) run, baseline or with a perturbation window.

`python train_cliphba.py --config pipeline/configs/training_config.yaml` trains the
seed-1 baseline into `<save_path>/baseline_seed1/`. A perturbed run sets
perturb_type / perturb_epoch / perturb_length / perturb_seed plus
baseline_checkpoint_path (see run_cliphba_perturbation.py) and lands in
`<save_path>/<type>_perturb_seed<P>/epoch<E>_length<L>/`.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import os
import shutil
import yaml

EXP_DIR    = Path(__file__).resolve().parent          # pipeline/scripts/training
PIPELINE   = EXP_DIR.parent.parent                    # pipeline/
REPO_ROOT  = PIPELINE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations.training.trainer import run_training_experiment
from temporal_perturbations.utils.parse_config_cli import parse_config_cli
from temporal_perturbations.utils.load_yaml_config import load_yaml_config
from temporal_perturbations.utils.save_config import save_config

# Keys that should be coerced to numeric values when loading the YAML config
TRAINING_INT_KEYS = [
    "epochs",
    "batch_size",
    "num_classes",
    "rank",
    "vision_layers",
    "transformer_layers",
    "early_stopping_patience",
    "random_seed",
    "cuda",
    "start_epoch",
    "end_epoch",
]
TRAINING_FLOAT_ONLY = ["lr", "train_portion", "weight_decay", "momentum"]


def _write_resolved_config(resolved_cfg: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        yaml.safe_dump(resolved_cfg, handle)


RUN_DIR_RE = re.compile(r"epoch(\d+)_length(\d+)")


def _find_previous_run_with_same_start(run_root: Path, start_epoch: int, current_length: int):
    """Return the longest shorter-length sibling run with exactly this start epoch, or (None, None)."""
    parent = run_root.parent
    if not parent.exists():
        return None, None

    candidates = []
    for entry in parent.iterdir():
        match = RUN_DIR_RE.fullmatch(entry.name)
        if not entry.is_dir() or not match:
            continue
        onset, length = int(match.group(1)), int(match.group(2))
        if onset == start_epoch and length < current_length:
            candidates.append((length, entry))

    if not candidates:
        return None, None
    best_length, best_dir = max(candidates, key=lambda t: t[0])
    return best_dir, best_length


def _find_checkpoint_dir(base_dir: Path, epoch: int):
    """
    Return the directory that actually contains
    ``dora_params/epoch{epoch}_dora_params.pth`` and the matching
    ``random_states/epoch{epoch}_random_states.pth``.

    Handles two layouts:
      - flat:   base_dir/dora_params/...   (Design A)
      - nested: base_dir/{run_name}/dora_params/...  (Design B, run_name
                appended by trainer.py)
    Returns the checkpoint directory and training_res path, or (None, None).
    """
    def _valid(d: Path) -> bool:
        return (
            (d / "dora_params" / f"epoch{epoch}_dora_params.pth").exists()
            and (d / "random_states" / f"epoch{epoch}_random_states.pth").exists()
        )

    if _valid(base_dir):
        return base_dir, base_dir / "training_res.csv"

    # One level deeper: base_dir/{run_name}/
    if base_dir.is_dir():
        for sub in sorted(base_dir.iterdir()):
            if sub.is_dir() and _valid(sub):
                return sub, sub / "training_res.csv"

    return None, None


def _latest_own_checkpoint(run_root: Path):
    """
    Find this run's OWN latest checkpoint (an interrupted or early-truncated
    prior attempt) that has both a DoRA checkpoint and a matching random-state
    file. Handles flat (run_root/dora_params/...) and nested
    (run_root/{run_name}/dora_params/...) layouts.

    Returns (checkpoint_dir, max_epoch) or (None, None).
    """
    def _scan(d: Path):
        dora_dir = d / "dora_params"
        rs_dir = d / "random_states"
        if not dora_dir.is_dir() or not rs_dir.is_dir():
            return None
        epochs = []
        for f in dora_dir.glob("epoch*_dora_params.pth"):
            try:
                ep = int(f.name[len("epoch"):].split("_")[0])
            except ValueError:
                continue
            if (rs_dir / f"epoch{ep}_random_states.pth").exists():
                epochs.append(ep)
        return max(epochs) if epochs else None

    if not run_root.is_dir():
        return None, None
    best = _scan(run_root)
    if best is not None:
        return run_root, best
    for sub in sorted(run_root.iterdir()):
        if sub.is_dir():
            ep = _scan(sub)
            if ep is not None:
                return sub, ep
    return None, None


def _resume_from_previous(config: dict, run_root: Path) -> dict:
    """
    If a previous run with the same start epoch exists in the save_path
    parent directory (and has a shorter perturb_length), reuse it to resume.
    """
    # Respect an explicitly-provided resume checkpoint (e.g. resuming this run's
    # own truncated/interrupted attempt) — don't override it with a sibling.
    if config.get("resume_checkpoint_path") and config.get("resume_from_epoch") is not None:
        return config

    perturb_epoch = int(config.get("perturb_epoch", 0))
    perturb_length = int(config.get("perturb_length", 0))

    # Only applicable for perturbation runs with length > 0
    if perturb_length <= 0:
        return config

    prev_dir, prev_length = _find_previous_run_with_same_start(
        run_root, start_epoch=perturb_epoch, current_length=perturb_length
    )
    if not prev_dir or prev_length is None:
        return config

    resume_from_epoch = perturb_epoch + prev_length - 1
    checkpoint_dir, prev_training_res = _find_checkpoint_dir(prev_dir, resume_from_epoch)

    if checkpoint_dir is None:
        print(f"[resume] epoch{perturb_epoch}_length{perturb_length}: sibling {prev_dir.name} has no "
              f"checkpoint at epoch {resume_from_epoch}; falling back to the baseline checkpoint")
        return config

    print(f"[resume] epoch{perturb_epoch}_length{perturb_length}: resuming from sibling "
          f"{prev_dir.name} at epoch {resume_from_epoch} ({checkpoint_dir})")
    config["resume_checkpoint_path"] = str(checkpoint_dir)
    config["resume_from_epoch"] = resume_from_epoch

    dest_training_res = run_root / "training_res.csv"
    if prev_training_res is not None and prev_training_res.exists():
        dest_training_res.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prev_training_res, dest_training_res)
        config["previous_training_res_path"] = str(prev_training_res)
    return config


def _prepare_single_run(config: dict, config_path: Path) -> dict:
    """
    Validate a single training run (baseline or perturbation) and ensure paths are set.

    
    """
    base_save_path = config.get("save_path")
    if not base_save_path:
        raise ValueError("Config must set 'save_path'.")

    training_mode = config.get("training_mode")

    # Validate mode-specific required fields
    if training_mode == "scratch":
        if not config.get("img_dir"):
            raise ValueError("Config must set 'img_dir' for training_mode='scratch'.")
        if config.get("perturb_type") == "random_target":
            raise ValueError(
                "perturb_type='random_target' requires continuous embedding targets; "
                "use training_mode='finetune' for this perturbation type."
            )
    elif training_mode == "finetune":
        if not config.get("img_annotations_file"):
            raise ValueError(
                "Config must set 'img_annotations_file' for training_mode='finetune'."
            )

    perturb_type = str(config.get("perturb_type") or "none")
    random_seed = config.get("random_seed")
    perturb_seed = config.get("perturb_seed")
    # Default to zeroed perturb settings for baseline (none) runs
    perturb_epoch = int(config.get("perturb_epoch") or 0)
    perturb_length = int(config.get("perturb_length") or 0)
    perturb_seed = int(perturb_seed if perturb_seed is not None else (random_seed or 0))

    if perturb_type and perturb_type.lower() != "none":
        # Include epoch/length tokens for perturbation runs
        seed_suffix = f"perturb_seed{perturb_seed}"
        run_root = Path(base_save_path) / f"{perturb_type}_{seed_suffix}" / f"epoch{perturb_epoch}_length{perturb_length}"
    else:
        seed_suffix = f"baseline_seed{random_seed}"
        run_root = Path(base_save_path) / seed_suffix
    run_root.mkdir(parents=True, exist_ok=True)
    save_path = run_root
    if perturb_length < 0:
        raise ValueError("perturb_length must be non-negative")
    if perturb_epoch > 0 and not config.get("baseline_checkpoint_path"):
        raise ValueError("baseline_checkpoint_path is required when perturb_epoch > 0")

    config["perturb_epoch"] = perturb_epoch
    config["perturb_length"] = perturb_length
    config["save_path"] = str(save_path)

    # Resume precedence:
    #   1. This run's OWN latest checkpoint (interrupted or early-truncated attempt)
    #   2. A shorter-length sibling with the same start epoch
    #   3. The baseline checkpoint (handled later in handle_checkpoint_resumption)
    own_dir, own_epoch = _latest_own_checkpoint(save_path)
    if own_dir is not None:
        config["resume_checkpoint_path"] = str(own_dir)
        config["resume_from_epoch"] = own_epoch
    config = _resume_from_previous(config, save_path)

    # Save the input config alongside the run artifacts
    save_config(config_path, save_path)
    _write_resolved_config(config, save_path / "resolved_config.yaml")
    return config


def main():
    """Entrypoint: load config specified via CLI and launch a single run."""
    os.chdir(REPO_ROOT)
    args = parse_config_cli("Train one CLIP-HBA run.", default_config=PIPELINE / "configs" / "training_config.yaml")
    config_path = Path(args.config)
    config = load_yaml_config(
        config_path,
        numeric_keys=TRAINING_INT_KEYS,
        float_only=TRAINING_FLOAT_ONLY,
    )
    if args.no_wandb:
        config['no_wandb'] = True
    if args.random_seed is not None:
        config['random_seed'] = args.random_seed
    if args.tag:
        config['run_tag'] = args.tag
    if args.save_path:
        config['save_path'] = args.save_path
    if args.img_dir:
        config['img_dir'] = args.img_dir

    # Regardless of experiment_type, we run a single training job; perturbation
    # behavior is controlled by perturb_epoch/perturb_length.
    prepared_config = _prepare_single_run(config, config_path)

    if args.dry_run:
        print("--- DRY RUN: resolved config ---")
        for key, value in prepared_config.items():
            print(f"  {key}: {value}")
        print(f"--- save_path: {prepared_config['save_path']} ---")
        return

    # Idempotency is handled by the resume logic above; --force is accepted for
    # CLI uniformity but not consumed.
    run_training_experiment(prepared_config)


if __name__ == "__main__":
    main()
