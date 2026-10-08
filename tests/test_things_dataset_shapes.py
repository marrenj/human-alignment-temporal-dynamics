from pathlib import Path

import pandas as pd
import torch
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset


def _write_rgb_png(path: Path, size=(10, 10), color=(128, 128, 128)) -> None:
    img = Image.new("RGB", size, color)
    img.save(path)


def test_things_behavioral_dataset_returns_expected_shapes(tmp_path: Path) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    _write_rgb_png(img_dir / "img0.png")

    # ThingsBehavioralDataset reads with index_col=0 and expects:
    # col0 = image filename, cols 1.. = float target dims.
    df = pd.DataFrame({"image": ["img0.png"], "t0": [0.1], "t1": [0.2]})
    csv_path = tmp_path / "annotations.csv"
    df.to_csv(csv_path)  # creates an index column automatically

    ds = ThingsBehavioralDataset(img_annotations_file=str(csv_path), img_dir=str(img_dir))
    image_name, image, targets = ds[0]

    assert image_name == "img0.png"
    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (3, 224, 224)
    assert torch.isfinite(image).all()

    assert isinstance(targets, torch.Tensor)
    assert targets.dtype == torch.float32
    assert tuple(targets.shape) == (2,)