"""Config types for the supervised ImageNet-1K arm of the perturbation matrix.

Parallels the CLIP config path (config/config.py + training_config.py) but for plain
N-way classification: a SupervisedModelConfig (timm arch + head) instead of a
CLIPConfig, and cross-entropy instead of contrastive loss. get_supervised_config()
mirrors config.get_config(): it reads config/trainingconfig/<name>.json (which embeds
a "model" block) and returns (training_config, model_config).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .training_config import TrainingConfig, DataConfig, OptimizerGroupConfig, DistributedConfig, WandbConfig, SchedulerConfig
TRAINING_CONFIG_DIR = Path(__file__).parent / "trainingconfig"


class SupervisedModelConfig(BaseModel):
    name: str                            # logical arch name: vit_b_16 | convnext_b | resnet50
    num_classes: int = 1000
    image_size: int = 224
    pretrained: bool = False             # train from scratch on ImageNet-1K
    drop_path_rate: float = 0.0


class SupervisedLossConfig(BaseModel):
    name: str = "cross_entropy"
    label_smoothing: float = 0.1


class ImageNetDataConfig(DataConfig):
    dataset_type: str = "imagenet1k"
    data_dir: Optional[str] = None       # LitData-indexed train split (HF imagenet-1k parquet shards)
    val_data_dir: Optional[str] = None   # LitData-indexed val split
    image_size: int = 224
    batch_size: int = 256                # per-GPU
    num_workers: int = 8
    pin_memory: bool = True
    # CIFAR caches for the optional --multi_val comparison suite.
    val_dataset_root: Optional[str] = None
    eval_batch_size: Optional[int] = 256

    # --- Augmentation (DeiT / ConvNeXt recipe) ---
    # RandAugment policy string (timm format). Common: "rand-m9-mstd0.5-inc1".
    # None = no RandAugment (only RandomResizedCrop + HorizontalFlip).
    randaugment_config: Optional[str] = None
    # MixUp + CutMix params; both 0 disables mixup entirely.
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"            # "batch" | "pair" | "elem"


class SupervisedImageNetTrainingConfig(TrainingConfig):
    """Supervised ImageNet-1K training config (cross-entropy, top-1/top-5)."""
    task: str = "supervised"
    data: DataConfig = Field(default_factory=ImageNetDataConfig)
    optimizer: OptimizerGroupConfig = Field(
        default_factory=lambda: OptimizerGroupConfig(
            name="adamw", lr=4e-3, weight_decay=0.05, beta1=0.9, beta2=0.999, eps=1e-8
        )
    )
    loss: SupervisedLossConfig = Field(default_factory=SupervisedLossConfig)  # type: ignore[assignment]
    epochs: int = 90
    monitor_metric: str = "val/imagenet/acc1"
    monitor_mode: str = "max"
    precision: str = "bf16-mixed"
    deterministic: bool = True


def load_supervised_training_config(config_name: str) -> tuple[SupervisedImageNetTrainingConfig, SupervisedModelConfig]:
    """Load a supervised preset JSON (embeds a top-level "model" block)."""
    config_file = TRAINING_CONFIG_DIR / f"{config_name}.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Supervised training config not found: {config_file}")
    cfg = json.loads(config_file.read_text())

    model_cfg = SupervisedModelConfig(**cfg.pop("model"))

    data_cfg = ImageNetDataConfig(**cfg.pop("data")) if cfg.get("data") else ImageNetDataConfig()
    opt_cfg = OptimizerGroupConfig(**cfg.pop("optimizer")) if cfg.get("optimizer") else None
    loss_cfg = SupervisedLossConfig(**cfg.pop("loss")) if cfg.get("loss") else SupervisedLossConfig()
    dist_cfg = DistributedConfig(**cfg.pop("distributed")) if cfg.get("distributed") else DistributedConfig()
    wandb_cfg = WandbConfig(**cfg.pop("wandb")) if cfg.get("wandb") else WandbConfig()
    sched_cfg = SchedulerConfig(**cfg.pop("scheduler")) if cfg.get("scheduler") else SchedulerConfig()
    cfg.pop("name", None)

    kwargs = {"data": data_cfg, "loss": loss_cfg, "distributed": dist_cfg,
              "wandb": wandb_cfg, "scheduler": sched_cfg, **cfg}
    if opt_cfg is not None:
        kwargs["optimizer"] = opt_cfg
    training_cfg = SupervisedImageNetTrainingConfig(**kwargs)
    return training_cfg, model_cfg


# Preset name -> True, so train_from_scratch.py can detect the supervised path.
SUPERVISED_PRESETS = {
    "supervised_imagenet_vitb16",
    "supervised_imagenet_convnext_b",
    "supervised_imagenet_resnet50",
}


def is_supervised_preset(name: str) -> bool:
    return name in SUPERVISED_PRESETS
