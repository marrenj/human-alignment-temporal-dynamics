"""Run RSA / triplet inference against a single trained checkpoint via a YAML config."""

from importlib import import_module
from pathlib import Path
import sys

EXP_DIR    = Path(__file__).resolve().parent          # pipeline/scripts/evaluation
PIPELINE   = EXP_DIR.parent.parent                    # pipeline/
REPO_ROOT  = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations.utils.load_yaml_config import load_yaml_config
from temporal_perturbations.utils.parse_config_cli import parse_config_cli
from temporal_perturbations.utils.save_config import save_config


INFERENCE_INT_KEYS = [
    "epoch",                       # eval epoch tag
    "epochs",
    "batch_size",
    "num_classes",
    "rank",
    "vision_layers",
    "transformer_layers",
    "random_seed",
    "cuda",
    "num_workers",
    "max_images_per_category",
]

INFERENCE_FLOAT_KEYS: list = []

# Bool keys must be coerced from "true"/"false" strings (env-passed) to real bools.
INFERENCE_BOOL_KEYS = [
    "pretrained",
    "load_hba",
    "cka_unbiased",
    "force",
]


def _coerce_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Cannot coerce {value!r} to bool")


def _apply_cli_overrides(config: dict, overrides: list) -> dict:
    """Apply ``--set key=value`` CLI overrides, coercing numerics and booleans."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set argument '{item}'. Expected format: key=value")
        key, _, value = item.partition("=")
        key = key.strip()
        if key in INFERENCE_INT_KEYS:
            config[key] = int(value)
        elif key in INFERENCE_FLOAT_KEYS:
            config[key] = float(value)
        elif key in INFERENCE_BOOL_KEYS:
            config[key] = _coerce_bool(value)
        else:
            config[key] = value
    return config


def main():
    """Entrypoint: load config specified via CLI and launch inference."""
    args = parse_config_cli("Run inference with an external config.")
    config_path = Path(args.config)
    config = load_yaml_config(
        config_path,
        numeric_keys=INFERENCE_INT_KEYS,
    )

    overrides = getattr(args, "overrides", None)
    if overrides:
        config = _apply_cli_overrides(config, overrides)

    # Resolve relative paths against cwd (the repo root where the script is invoked),
    # not the config file's directory, so a config copied elsewhere still works.
    cwd = Path.cwd()
    def _resolve_path(p: str | Path | None) -> str:
        if p is None:
            return None
        pth = Path(p)
        return str(pth if pth.is_absolute() else (cwd / pth))

    for key in ("inference_save_dir", "model_weights_path", "img_dir", "annotations_file"):
        if key in config and config[key]:
            config[key] = _resolve_path(config[key])

    if "reference_rdm_paths" in config and config["reference_rdm_paths"]:
        config["reference_rdm_paths"] = {
            roi: _resolve_path(path) for roi, path in config["reference_rdm_paths"].items()
        }

    if args.tag:
        config['run_tag'] = args.tag
    if args.save_path:
        config['inference_save_dir'] = args.save_path
    if args.img_dir:
        config['img_dir'] = args.img_dir

    if args.dry_run:
        print("--- DRY RUN: resolved inference config ---")
        for key, value in config.items():
            print(f"  {key}: {value}")
        return

    save_config(config_path, config["inference_save_dir"])
    # --force is accepted for CLI uniformity: run_inference overwrites results per call.
    inference_module = import_module("temporal_perturbations.inference.inference_core")
    run_inference = inference_module.run_inference
    run_inference(config)


if __name__ == "__main__":
    main()