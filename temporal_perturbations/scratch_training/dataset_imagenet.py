"""ImageNet-1K LitData streaming dataset + LightningDataModule.

Mirrors dataset.py's YFCC15M streaming layout but for HuggingFace `imagenet-1k`
parquet shards: each row has `image` (jpeg bytes, possibly wrapped in {"bytes":..., "path":...})
and `label` (int 0-999). The HF repo is gated; download it once and build a LitData index
per split (`litdata.index_parquet_dataset`), then pass the two split directories as
`data.data_dir` / `data.val_data_dir` (CLI: --imagenet_train_dir / --imagenet_val_dir).

Train: standard 224 RandomResizedCrop + flip + ImageNet normalize.
Val:   Resize(256) + CenterCrop(224) + ImageNet normalize.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

import litdata as ld
from PIL import Image
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms as T
from litdata.streaming.item_loader import ParquetLoader

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224, interpolation: str = "bicubic",
                          randaugment_config: Optional[str] = None) -> T.Compose:
    """Train-time image transform. When randaugment_config is provided (e.g.
    "rand-m9-mstd0.5-inc1" — the DeiT/ConvNeXt default), the RandAugment policy
    is inserted before ToTensor/Normalize. ResNet50 typically leaves this None."""
    interp = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR}[interpolation]
    ops = [
        T.RandomResizedCrop(image_size, interpolation=interp),
        T.RandomHorizontalFlip(),
    ]
    if randaugment_config:
        from timm.data.auto_augment import rand_augment_transform
        ra = rand_augment_transform(
            randaugment_config,
            hparams={"translate_const": int(image_size * 0.45),
                     "img_mean": tuple(int(255 * x) for x in IMAGENET_MEAN)},
        )
        ops.append(ra)
    ops.extend([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return T.Compose(ops)


def build_val_transform(image_size: int = 224, interpolation: str = "bicubic") -> T.Compose:
    interp = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR}[interpolation]
    resize = int(image_size * 256 / 224)
    return T.Compose([
        T.Resize(resize, interpolation=interp),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _decode_image(value: Any) -> Image.Image:
    """HF imagenet-1k parquet stores `image` as either raw bytes or {bytes,path} dict."""
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Cannot decode image of type {type(value)}")


class ImageNetLitDataDataset(ld.StreamingDataset):
    """StreamingDataset over imagenet-1k parquet shards."""

    def __init__(self, *args, transform, image_col: str = "image", label_col: str = "label", **kwargs):
        super().__init__(*args, **kwargs)
        # Mirror YFCC15MLitDataDataset: don't pass transform up; the base class would
        # apply it to the raw dict instead of the decoded PIL image.
        self._transform = transform
        self.transform = lambda x: x
        self.image_col = image_col
        self.label_col = label_col
        self._last_good_sample: Optional[tuple] = None
        self._bad = 0
        self._warned_chunks: set = set()

    def __getitem__(self, index):
        try:
            raw = super().__getitem__(index)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            self._bad += 1
            chunk = getattr(index, "chunk_index", None)
            if chunk not in self._warned_chunks:
                self._warned_chunks.add(chunk)
                logger.warning(f"ImageNet read error chunk={chunk} idx={index}: {type(e).__name__}: {e}")
            return self._last_good_sample

        try:
            img = _decode_image(raw[self.image_col])
            tensor = self._transform(img)
            label = int(raw[self.label_col])
        except Exception as e:
            self._bad += 1
            chunk = getattr(index, "chunk_index", None)
            if chunk not in self._warned_chunks:
                self._warned_chunks.add(chunk)
                logger.warning(f"ImageNet decode error chunk={chunk}: {type(e).__name__}: {e}")
            return self._last_good_sample

        sample = (tensor, label)
        self._last_good_sample = sample
        return sample


class ImageNetDataModule(pl.LightningDataModule):
    """Defers dataloader construction to setup() so DDP is initialized first."""

    def __init__(self, config, seed: int = 42, distributed: bool = True):
        super().__init__()
        self.config = config
        self.seed = seed
        self.distributed = distributed
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_loader is not None:
            return

        train_dir = self.config.parquet_data_dir or self.config.data_dir
        val_dir = self.config.val_data_dir
        if not train_dir:
            raise ValueError("ImageNet train split not set (--imagenet_train_dir / config.data.data_dir)")
        if not val_dir:
            raise ValueError("ImageNet val split not set (--imagenet_val_dir / config.data.val_data_dir)")

        train_tfm = build_train_transform(
            self.config.image_size,
            self.config.interpolation,
            randaugment_config=getattr(self.config, "randaugment_config", None),
        )
        val_tfm = build_val_transform(self.config.image_size, self.config.interpolation)

        # ParquetLoader is required for raw HF parquet shards — litdata's default item_loader
        # expects its own optimized binary chunk format and errors with
        # "Please, use Cache(..., item_loader=ParquetLoader(...))" otherwise.
        train_ds = ImageNetLitDataDataset(
            input_dir=train_dir, transform=train_tfm, shuffle=True, seed=self.seed, drop_last=True,
            item_loader=ParquetLoader(),
        )
        val_ds = ImageNetLitDataDataset(
            input_dir=val_dir, transform=val_tfm, shuffle=False, seed=self.seed, drop_last=False,
            item_loader=ParquetLoader(),
        )

        self._train_loader = ld.StreamingDataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )
        self._val_loader = ld.StreamingDataLoader(
            val_ds,
            batch_size=self.config.eval_batch_size or self.config.batch_size,
            num_workers=max(2, self.config.num_workers // 2),
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader
