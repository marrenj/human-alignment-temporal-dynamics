from __future__ import annotations

from pathlib import Path
import shutil


def save_config(src: Path | str, destination_dir: Path | str | None) -> None:
    """
    Copy the configuration file ``src`` into ``destination_dir`` for provenance.

    Parameters
    ----------
    src:
        Path to the original YAML config file.
    destination_dir:
        Directory where the config copy should be stored. When ``None`` or empty,
        the function is a no-op.
    """
    if not destination_dir:
        return

    src_path = Path(src)
    dest_dir = Path(destination_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    destination = dest_dir / src_path.name
    try:
        if src_path.resolve() == destination.resolve():
            return
    except FileNotFoundError:
        # One of the paths may not exist yet; go ahead with the copy.
        pass

    shutil.copy2(src_path, destination)
