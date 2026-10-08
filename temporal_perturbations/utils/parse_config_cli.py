from __future__ import annotations

import argparse
from pathlib import Path

def parse_config_cli(description: str, default_config: Path | None = None) -> argparse.Namespace:
    """Build the standard CLI for scripts that take a single --config YAML file."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        required=default_config is None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging entirely.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Override random_seed from the config (e.g. for multi-seed parallel runs).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional run tag to append to checkpoint/output paths.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Override save_path in the YAML config (e.g. for repo-relative output dirs).",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Override img_dir in the YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Bypass idempotency cache and re-run even if outputs exist.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Print the resolved config and planned actions without executing.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config key. Repeatable: --set key1=v1 --set key2=v2",
    )
    return parser.parse_args()