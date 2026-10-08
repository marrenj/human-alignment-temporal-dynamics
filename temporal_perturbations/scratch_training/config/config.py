import json
from pathlib import Path
from typing import Dict, Any, Optional

from .model_config import (
    CLIPConfig,
    CLIPVisionConfig,
    CLIPTextConfig,
)
from .training_config import (
    DataConfig,
    OptimizerGroupConfig,
    DualOptimizerConfig,
    SchedulerConfig,
    ContrastiveLossConfig,
    WandbConfig,
    DistributedConfig,
    TrainingConfig,
    YFCC15MLitDataConfig,
    DeCLIPSchedulerConfig,
    CLIPLossConfig,
    DeCLIPLitDataTrainingConfig,
)
CONFIG_DIR = Path(__file__).parent
MODEL_CONFIG_DIR = CONFIG_DIR / "modelconfig"
TRAINING_CONFIG_DIR = CONFIG_DIR / "trainingconfig"


def load_json_config(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


def load_model_config(model_name: str) -> CLIPConfig:
    config_file = MODEL_CONFIG_DIR / f"{model_name}.json"
    config_data = load_json_config(config_file)

    vision_cfg = CLIPVisionConfig(**config_data["vision_cfg"])
    text_cfg = CLIPTextConfig(**config_data["text_cfg"])

    config_data_copy = config_data.copy()
    config_data_copy.pop("vision_cfg")
    config_data_copy.pop("text_cfg")
    config_data_copy.pop("name", None)

    return CLIPConfig(vision_cfg=vision_cfg, text_cfg=text_cfg, **config_data_copy)


def load_training_config(config_name: str) -> TrainingConfig:
    config_file = TRAINING_CONFIG_DIR / f"{config_name}.json"
    config_data = load_json_config(config_file)
    config_type = config_data.get("name", config_name)

    data_config = None
    if config_data.get("data"):
        data_dict = config_data["data"]
        if data_dict.get("dataset_type") == "yfcc15m_litdata":
            data_config = YFCC15MLitDataConfig(**data_dict)
        else:
            data_config = DataConfig(**data_dict)

    optimizer_config = None
    if config_data.get("optimizer"):
        opt_dict = config_data["optimizer"]
        if "vision" in opt_dict and "text" in opt_dict:
            optimizer_config = DualOptimizerConfig(
                vision=OptimizerGroupConfig(**opt_dict["vision"]),
                text=OptimizerGroupConfig(**opt_dict["text"]),
            )
        else:
            optimizer_config = OptimizerGroupConfig(**opt_dict)

    scheduler_config = None
    if config_data.get("scheduler"):
        sched_dict = config_data["scheduler"]
        if "warmup_lr_vision" in sched_dict or "warmup_lr_text" in sched_dict:
            scheduler_config = DeCLIPSchedulerConfig(**sched_dict)
        else:
            scheduler_config = SchedulerConfig(**sched_dict)

    loss_config = None
    if config_data.get("loss"):
        loss_dict = config_data["loss"]
        if loss_dict.get("margin") is not None:
            loss_config = ContrastiveLossConfig(**loss_dict)
        else:
            loss_config = CLIPLossConfig(**loss_dict)

    distributed_config = None
    if config_data.get("distributed"):
        distributed_config = DistributedConfig(**config_data["distributed"])

    wandb_config = None
    if config_data.get("wandb"):
        wandb_config = WandbConfig(**config_data["wandb"])

    config_data_copy = config_data.copy()
    for key in ["data", "optimizer", "scheduler", "loss", "distributed", "wandb", "name", "model_config"]:
        config_data_copy.pop(key, None)

    if "declip" in config_type.lower() or "yfcc" in config_type.lower():
        return DeCLIPLitDataTrainingConfig(
            data=data_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            distributed=distributed_config,
            wandb=wandb_config,
            **config_data_copy,
        )
    return TrainingConfig(
        data=data_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        loss=loss_config,
        distributed=distributed_config,
        wandb=wandb_config,
        **config_data_copy,
    )


def get_config(training_config_name: str, model_config_name: Optional[str] = None) -> tuple[TrainingConfig, CLIPConfig]:
    training_config = load_training_config(training_config_name)

    if model_config_name is None:
        config_file = TRAINING_CONFIG_DIR / f"{training_config_name}.json"
        config_data = load_json_config(config_file)
        model_config_name = config_data.get("model_config")
        if model_config_name is None:
            raise ValueError(
                f"No model_config specified in training config {training_config_name} and no model_config_name provided"
            )

    model_config = load_model_config(model_config_name)
    return training_config, model_config


def list_available_configs() -> Dict[str, list]:
    model_configs = []
    training_configs = []
    if MODEL_CONFIG_DIR.exists():
        model_configs = [f.stem for f in MODEL_CONFIG_DIR.glob("*.json")]
    if TRAINING_CONFIG_DIR.exists():
        training_configs = [f.stem for f in TRAINING_CONFIG_DIR.glob("*.json")]
    return {"model_configs": sorted(model_configs), "training_configs": sorted(training_configs)}


def get_vit_b_32_config() -> CLIPConfig:
    return load_model_config("vit_b_32")


def get_resnet_50_config() -> CLIPConfig:
    return load_model_config("resnet_50")


def get_baseline_yfcc15m_vitb32_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    return get_config("declip_yfcc15m_litdata_vitb32")


def get_baseline_yfcc15m_resnet50_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    return get_config("declip_yfcc15m_litdata_resnet50")


VIT_B_32_CONFIG = get_vit_b_32_config()
RESNET_50_CONFIG = get_resnet_50_config()
BASELINE_YFCC15M_VITB32_CONFIG, _ = get_baseline_yfcc15m_vitb32_config()
BASELINE_YFCC15M_RESNET50_CONFIG, _ = get_baseline_yfcc15m_resnet50_config()


if __name__ == "__main__":
    print("Available configurations:")
    configs = list_available_configs()
    print(f"Model configs: {configs['model_configs']}")
    print(f"Training configs: {configs['training_configs']}")
    training_config, model_config = get_config("declip_yfcc15m_litdata_vitb32")
    print(f"Loaded training config: {training_config.wandb.tags}")
    print(f"Loaded model config: {model_config.vision_cfg.patch_size}")
