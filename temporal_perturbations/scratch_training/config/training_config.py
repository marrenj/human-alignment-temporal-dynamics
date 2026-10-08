from pydantic import BaseModel, Field
from typing import Optional, List, Union, Tuple, Dict, Any, Literal
from typing import ClassVar



class DataConfig(BaseModel):
    dataset_type: str = "yfcc15m_litdata"
    dataset_root: Optional[str] = None
    hf_dataset_name: Optional[str] = None  # e.g. "Kaichengalex/YFCC15M"
    hf_cache_dir: Optional[str] = None
    hf_streaming: bool = False

    # --- LitData ---
    use_litdata: bool = True
    parquet_data_dir: Optional[str] = None

    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    interpolation: str = "bicubic"
    tokenizer_name: str = "openai/clip-vit-base-patch32"
    image_col: str = "images"
    text_col: str = "texts"
    val_dataset_root: Optional[str] = None
    eval_batch_size: Optional[int] = None

class OptimizerConfig(BaseModel):
    name: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6

class OptimizerGroupConfig(BaseModel):
    name: Literal["adamw", "sgd", "adam"]
    lr: float
    weight_decay: float
    momentum: Optional[float] = None  # only used by SGD
    nesterov: Optional[bool] = None   # only used by SGD; requires momentum > 0
    beta1: Optional[float] = None     # only for AdamW
    beta2: Optional[float] = None
    eps: Optional[float] = None
    amsgrad: Optional[bool] = None


class DualOptimizerConfig(BaseModel):
    """
    Dual Optimizer Config for Vision and Text for DeCLIP like training
    """
    vision: OptimizerGroupConfig = OptimizerGroupConfig(
        name="adamw",
        lr=1e-3,
        weight_decay=0.05,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        amsgrad=False,
    )
    text: OptimizerGroupConfig = OptimizerGroupConfig(
        name="sgd",
        lr=0.02,
        weight_decay=1e-4,
        momentum=0.9,
    )

class SchedulerConfig(BaseModel):
    name: str = "cosine"
    warmup_steps: Optional[int] = 2500
    epochs: Optional[int] = None
    warmup_lr_vision: Optional[float] = 1e-4   # For vision encoder (AdamW)
    warmup_lr_text: Optional[float] = 1e-3     # For text encoder (SGD)
    # For const-cooldown
    lr_cooldown_end: Optional[float] = 1e-7
    lr_cooldown_power: Optional[float] = 1.0
    cooldown_steps: Optional[int] = None
    min_lr: Optional[float] = 0.0

class ContrastiveLossConfig(BaseModel):
    margin: float = 0.1 
    lr: float = 1e-4 
    distance_type: str = "cosine" # "cosine" or "euclidean"
    # For CLIP InfoNCE Loss
    local_loss: bool = False
    gather_with_grad: bool = False
    cache_labels: bool = False
    # Shared scale param?
    scale: float = 1.0 # Used by TripletLoss, logit_scale learned in CLIP


class WandbConfig(BaseModel):
    enabled: bool = False
    project: Optional[str] = None
    name: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List[str]] = ["clip", "vit-b32", "dreamsim"]
    log_model: bool = True
    save_code: bool = True


class DistributedConfig(BaseModel):
    strategy: str = "ddp"
    sync_batchnorm: bool = True
    find_unused_parameters: bool = False
    num_nodes: int = 1


class TrainingConfig(BaseModel):
    """Base Training Config Schema"""
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: Union[OptimizerGroupConfig, DualOptimizerConfig] = Field(default_factory=DualOptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: ContrastiveLossConfig = Field(default_factory=ContrastiveLossConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    epochs: Optional[int] = None # Presets should define this
    precision: str = "bf16-mixed" # Default to bf16-mixed
    seed: int = 42
    accelerator: str = "auto"
    devices: Union[int, str, List[int]] = "auto"
    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: Optional[int] = None
    log_every_n_steps: int = 50
    save_dir: Optional[str] = None
    save_top_k: int = 32
    save_every_n_train_steps: Optional[int] = None # DEPRECATED
    deterministic: bool = True
    freeze_vision_blocks: int = 0  # For continual learning (optional)
    
    validate_every_n_epoch: Optional[int] = 1  # Default: check every epoch (None to disable)
    validate_every_n_steps: Optional[Union[int, float]] = None  # Step-based validation (None to disable)
                                                                # Float (0-1) = fraction of epoch
                                                                # Int (>1) = number of steps
    
    monitor_metric: Optional[str] = None # Metric to monitor
    monitor_mode: str = "min"          # 'min' or 'max'
    target_batch_size: Optional[int] = None
    perturb_type: str = "none"   # none | uniform_images | image_noise | label_shuffle
    perturb_epoch: int = 0       # First epoch (0-indexed) to apply the perturbation
    perturb_length: int = 0      # Number of consecutive epochs to apply the perturbation
    perturb_seed: int = 42       # Seed for deterministic perturbations


# --- DeCLIP-like Config for YFCC15M ---
# Updated config for using LitData with local Parquet files
class YFCC15MLitDataConfig(DataConfig):
    dataset_type: str = "yfcc15m_litdata" # New type identifier
    use_litdata: bool = True
    # --- Set paths for your local data ---
    parquet_data_dir: Optional[str] = None # User provided path
    # --- Hugging Face specific fields are not used ---
    hf_dataset_name: Optional[str] = 'hf://datasets/Kaichengalex/YFCC15M/data'
    hf_cache_dir: Optional[str] = None
    hf_streaming: bool = False
    dataset_index_path: Optional[str] = None # None: use the parquet directory itself as the index path
    litdata_max_cache_size: Optional[int] = 200 * 1024 ** 3  # Max bytes for LitData chunk cache (~200 GB)
    # --- Other settings ---
    dataset_root: Optional[str] = None # Not used for parquet streaming
    image_col: str = "images" # Standard key in the parquet files
    text_col: str = "texts" # Standard key in the parquet files (adjust if different)
    interpolation: str = "bicubic"
    image_size: int = 224
    batch_size: int = 1024 # Matches previous update
    num_workers: int = 16 # From DeCLIP config
    pin_memory: bool = False # Often False with streaming/iterable datasets
    val_dataset_root: Optional[str] = None
    eval_batch_size: Optional[int] = 360 # Example smaller eval batch size

class DeCLIPOptimizerConfig(OptimizerConfig):
    name: str = "adamw"
    lr: float = 0.0005
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8


class DeCLIPSchedulerConfig(SchedulerConfig):
    name: str = "cosine"
    # Base LR taken from OptimizerConfig
    warmup_lr: float = 0.001 # Informational
    min_lr: float = 0.0 # Informational
    warmup_steps: int = 2500
    epochs: Optional[int] = None # Set by main config

class CLIPLossConfig(ContrastiveLossConfig):
    # Parameters specific to CLIP InfoNCE loss
    margin: Optional[float] = None # Not used
    distance_type: Optional[str] = None # Not used
    local_loss: bool = False
    gather_with_grad: bool = False
    cache_labels: bool = True


# Updated DeCLIP Training Config using LitData (Epoch Based)
class DeCLIPLitDataTrainingConfig(TrainingConfig):
    """Training config based on DeCLIP's YFCC15M ViT-B/32 setup, using LitData (Epoch Based)"""
    data: DataConfig = Field(default_factory=YFCC15MLitDataConfig)
    optimizer: Union[OptimizerGroupConfig, DualOptimizerConfig] = Field(default_factory=DualOptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=DeCLIPSchedulerConfig)
    loss: ContrastiveLossConfig = Field(default_factory=CLIPLossConfig)
    deterministic: bool = True
    epochs: int = 32
    validate_every_n_steps: Optional[Union[int, float]] = None
    save_every_n_train_steps: Optional[int] = None
    monitor_metric: str = "val/cifar100/epoch_acc1"
    monitor_mode: str = "max"
    wandb: WandbConfig = Field(default_factory=lambda: WandbConfig(tags=["clip", "vit-b32", "yfcc15m", "litdata"]))
    target_batch_size: Optional[int] = None
    save_dir: Optional[str] = None

