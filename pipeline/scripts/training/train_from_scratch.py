"""Train the from-scratch models (RN50 / ViT-B/16 / ConvNeXt-B on ImageNet-1K, CLIP ViT-B/32 on YFCC15M).

Every paper run's exact arguments are in `pipeline/configs/scratch_runs.yaml`
(run name -> args). The four families:

| family        | preset                          | data               | epochs | onsets (early / mid / late) |
| ------------- | ------------------------------- | ------------------ | ------ | --------------------------- |
| RN50          | supervised_imagenet_resnet50    | ImageNet-1K        | 90     | 5 / 30 / 70                 |
| ViT-B/16      | supervised_imagenet_vitb16      | ImageNet-1K        | 90     | 5 / 30 / 70                 |
| ConvNeXt-B    | supervised_imagenet_convnext_b  | ImageNet-1K        | 90     | 5 / 30 / 70                 |
| CLIP ViT-B/32 | declip_yfcc15m_litdata_vitb32   | YFCC15M (LitData)  | 32     | 5 / 13 / 28                 |

Procedure per family: train the unperturbed baseline with `--save_every_epoch`
(writes `<save_dir>/<run>/checkpoints/all_epochs/epoch_NNN.ckpt`). Each perturbed run
is a fork: `--resume_ckpt_path <baseline>/.../epoch_{E-1}.ckpt --perturb_type <type>
--perturb_epoch E --perturb_length 1`, so it is bit-identical to the baseline up to
onset E, sees one perturbed epoch, then trains unperturbed to the end of the same
schedule (`--epochs` is the total, so the LR schedule matches the baseline). The
perturbation types are image_noise, label_shuffle and uniform_images; all runs use
seed 42 (seed-43 replicate: `--seed 43`, run names suffixed `_seed43`).

Launch with `torchrun --nproc_per_node <gpus> train_from_scratch.py ...`; the GPU
count is part of the recipe because it sets the gradient-accumulation factor.
"""
import os
import sys
import argparse
import random
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import math

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

# Configure logging BEFORE applying the litdata patch so its INFO/WARNING messages
# (whether the patch wrapped its targets, or the targets were missing) actually appear.
logging.basicConfig(level=logging.INFO)

from temporal_perturbations.scratch_training.litdata_patch import apply as _apply_litdata_patch
_apply_litdata_patch()

# Cosmetic: Lightning/torch < matching-version don't register a device_id on the DDP process
# group, so the first barrier() call logs a one-shot hint. Functionally harmless — suppress
# before any distributed code runs.
warnings.filterwarnings(
    "ignore",
    message=r"barrier\(\): using the device under current context\..*",
)

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
)
from pytorch_lightning.strategies import DDPStrategy


class DefaultStreamDDPStrategy(DDPStrategy):
    """DDP strategy that initialises the DDP wrapper under the default CUDA stream.

    PyTorch's DDP registers AccumulateGrad hooks during DistributedDataParallel
    __init__.  If that call happens on a non-default stream (e.g. an internal
    NCCL setup stream), the resulting AccumulateGrad nodes live on that stream.
    Every subsequent forward/backward on the default stream then triggers the
    "AccumulateGrad node's stream does not match" UserWarning and an unnecessary
    cross-stream sync.  Wrapping the init in default_stream() eliminates the
    mismatch.
    """

    def _setup_model(self, model):
        if torch.cuda.is_available():
            device = torch.device("cuda", self.local_rank)
            with torch.cuda.stream(torch.cuda.default_stream(device)):
                return super()._setup_model(model)
        return super()._setup_model(model)
from pytorch_lightning.utilities.rank_zero import rank_zero_info # For logging on rank 0

from temporal_perturbations.scratch_training.config import (
    get_config,
    BASELINE_YFCC15M_VITB32_CONFIG,
    BASELINE_YFCC15M_RESNET50_CONFIG,
    TrainingConfig,
    DeCLIPLitDataTrainingConfig,
)
from temporal_perturbations.scratch_training.dataset import create_dataloaders, YFCC15MDataModule
from temporal_perturbations.scratch_training.lightning_module import CLIPLightningModule
from temporal_perturbations.scratch_training.supervised_module import SupervisedLightningModule
from temporal_perturbations.scratch_training.dataset_imagenet import ImageNetDataModule
from temporal_perturbations.scratch_training.config.supervised_config import load_supervised_training_config, is_supervised_preset

from temporal_perturbations import paths

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP model")

    parser.add_argument(
        "--config_preset", type=str, default="declip_yfcc15m_litdata_vitb32",
        choices=[
            "declip_yfcc15m_litdata_vitb32",
            "declip_yfcc15m_litdata_resnet50",
            "supervised_imagenet_vitb16",
            "supervised_imagenet_convnext_b",
            "supervised_imagenet_resnet50",
        ],
        help="Training preset (CLIP YFCC15M or supervised ImageNet-1K)."
    )

    parser.add_argument(
        "--test", action="store_true", default=False,
        help="Test mode: run only 5 training steps and 1 epoch to verify the training pipeline works."
    )

    # Dataset arguments override presets when provided.
    parser.add_argument(
        "--dataset_type", type=str, choices=["yfcc15m_litdata", "imagenet1k"],
        help="Type of dataset (yfcc15m_litdata for CLIP, imagenet1k for supervised). Overrides preset."
    )
    parser.add_argument(
        "--hf_dataset_name", type=str, help="HF dataset name (if dataset_type='yfcc15m')."
    )
    parser.add_argument(
        "--hf_cache_dir", type=str, help="HF cache directory."
    )
    parser.add_argument(
        "--hf_streaming", action="store_true", default=False, help="Enable HF streaming (if dataset_type='yfcc15m')."
    )
    parser.add_argument(
        "--use_litdata", action="store_true", default=None, help="Force use of LitData (if dataset_type='yfcc15m_litdata')."
    )
    parser.add_argument(
        "--parquet_data_dir", type=str, help="Path to local parquet files directory (for litdata). Overrides preset."
    )
    parser.add_argument(
        "--image_col", type=str, help="Column name for image data. Overrides preset."
    )
    parser.add_argument(
        "--text_col", type=str, help="Column name for text data. Overrides preset."
    )
    parser.add_argument(
        "--tokenizer_name", type=str, help="Tokenizer model name. Overrides preset."
    )
    parser.add_argument(
        "--imagenet_train_dir", type=str, help="LitData-indexed ImageNet-1K train parquet shards (supervised presets)."
    )
    parser.add_argument(
        "--imagenet_val_dir", type=str, help="LitData-indexed ImageNet-1K validation parquet shards (supervised presets)."
    )
    parser.add_argument(
        "--val_dataset_root", type=str, help="Root directory for validation dataset (e.g., CIFAR10, CIFAR100, etc.). Overrides preset."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, help="Specific batch size for evaluation dataloader. Overrides preset."
    )
    parser.add_argument(
        "--multi_val", action="store_true", default=False, help="Enable multiple validation datasets (CIFAR10 and CIFAR100)"
    )

    # Training arguments override presets when provided.
    parser.add_argument("--batch_size", type=int, help="Batch size per device. Overrides preset.")
    parser.add_argument("--num_workers", type=int, help="Dataloader workers. Overrides preset.")
    parser.add_argument("--epochs", type=int, help="Number of epochs. Overrides preset.")
    parser.add_argument("--lr", type=float, help="Learning rate. Overrides preset.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay. Overrides preset.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], help="Optimizer. Overrides preset.")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "const", "const-cooldown"], help="LR scheduler. Overrides preset.")
    parser.add_argument("--warmup_steps", type=int, help="LR warmup steps. Overrides preset.")
    parser.add_argument("--gradient_clip_val", type=float, help="Gradient clipping value. Overrides preset.")
    parser.add_argument("--accumulate_grad_batches", type=int, help="Gradient accumulation steps. Overrides preset.")
    parser.add_argument(
        "--validate_every_n_epoch", 
        type=int, 
        help="Run validation every N epochs (default=1, None to disable). Overrides preset. Takes priority over --validate_every_n_steps."
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=lambda v: int(v) if v.lstrip("-").isdigit() else float(v),
        help="Run validation every N steps or fraction of epoch (float 0-1 = fraction of epoch, int > 1 = steps). Overrides preset."
    )

    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (auto, gpu, cpu).")
    parser.add_argument("--devices", type=str, default="auto", help="Device indices (comma-separated), number of devices, or 'auto'.")

    parser.add_argument("--strategy", type=str, default=None, choices=["ddp", "ddp_spawn", "ddp_sharded", "deepspeed", None], help="Distributed training strategy.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training.")
    parser.add_argument("--sync_batchnorm", action="store_true", default=None, help="Enable sync batchnorm.")
    parser.add_argument("--no_sync_batchnorm", action="store_false", dest="sync_batchnorm", help="Disable sync batchnorm.")
    parser.add_argument("--find_unused_parameters", action="store_true", default=None, help="Find unused parameters in DDP.")
    parser.add_argument("--no_find_unused_parameters", action="store_false", dest="find_unused_parameters", help="Disable find unused parameters in DDP.")
    parser.add_argument("--deterministic", action="store_true", default=None, help="Use deterministic algorithms.")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic", help="Disable deterministic algorithms.")


    parser.add_argument("--save_dir", type=str, default=None, help="Root directory for saving checkpoints and logs (default: paths.yaml runs_root).")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name (default: auto-generated).")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log training metrics every N steps.")
    parser.add_argument("--save_top_k", type=int, help="Save top-k checkpoints based on monitor metric. Overrides preset.")
    parser.add_argument("--monitor_metric", type=str, default=None, help="Metric to monitor for checkpoints/early stopping. Overrides preset.")
    parser.add_argument("--monitor_mode", type=str, choices=["min", "max"], help="Mode for monitored metric ('min' or 'max'). Overrides preset.")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument(
        "--save_every_epoch", action="store_true", default=False,
        help="Save a checkpoint at the end of every epoch under <ckpt_dir>/all_epochs/epoch_NNN.ckpt. "
             "Required for fork-from-baseline perturbation runs that resume at specific epochs."
    )
    parser.add_argument(
        "--stop_epoch", type=int, default=None,
        help="Stop training after this epoch completes (0-indexed). Use with full --epochs "
             "so the LR schedule matches the original run while only training through the target epoch."
    )
    parser.add_argument(
        "--sub_epoch_ckpt_steps", type=str, default=None,
        help="Comma-separated list of batch indices at which to save sub-epoch checkpoints "
             "(e.g. '10,25,63,125,250,500,1000,1250'). Requires --sub_epoch_ckpt_epoch."
    )
    parser.add_argument(
        "--sub_epoch_ckpt_epoch", type=int, default=None,
        help="Epoch (0-indexed) during which to save sub-epoch checkpoints. "
             "Defaults to perturb_epoch + perturb_length (first recovery epoch) if perturbation is configured."
    )

    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging (off by default).")
    parser.add_argument("--wandb_project", type=str, help="W&B project name. Overrides preset.")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity name. Overrides preset.")
    parser.add_argument("--wandb_tags", type=str, help="W&B tags (comma-separated). Overrides preset.")
    parser.add_argument("--wandb_log_model", type=str, help="Log model checkpoints to W&B ('all', 'best', 'False'). Overrides preset.")
    parser.add_argument("--wandb_save_code", action="store_true", default=None, help="Save code to W&B.")
    parser.add_argument("--no_wandb_save_code", action="store_false", dest="wandb_save_code", help="Disable code saving to W&B.")

    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile (requires PyTorch >= 2.0).")
    parser.add_argument("--no_compile", action="store_false", dest="compile", help="Disable torch.compile.")

    parser.add_argument("--seed", type=int, help="Random seed. Overrides preset.")

    parser.add_argument(
        "--resume_ckpt_path", type=str, default=None,
        help="Path to a Lightning checkpoint to resume training from (restores model, optimizer, and epoch state)."
    )
    parser.add_argument(
        "--freeze_vision_blocks", type=int, default=None,
        help="Number of ViT blocks to freeze in the vision model (0 = none, >0 freezes from the start)."
    )

    parser.add_argument(
        "--perturb_type", type=str, default=None,
        choices=["none", "uniform_images", "image_noise", "label_shuffle"],
        help="Perturbation type to apply during training. Overrides preset."
    )
    parser.add_argument(
        "--perturb_epoch", type=int, default=None,
        help="First epoch (0-indexed) to apply the perturbation. Overrides preset."
    )
    parser.add_argument(
        "--perturb_length", type=int, default=None,
        help="Number of consecutive epochs to apply the perturbation. Overrides preset."
    )
    parser.add_argument(
        "--perturb_seed", type=int, default=None,
        help="Seed for deterministic perturbations. Overrides preset."
    )

    return parser.parse_args()

def count_available_gpus():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()

def detect_torchrun_environment():
    env_keys = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT']
    env_info = {}
    torchrun_detected = all(key in os.environ for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])
    if torchrun_detected:
        for key in env_keys:
            env_info[key] = os.environ.get(key)
    return torchrun_detected, env_info

def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
             logger.info("Setting deterministic CUDA options")
             torch.backends.cudnn.deterministic = True
             torch.backends.cudnn.benchmark = False
        else:
             logger.info("Setting non-deterministic CUDA options (benchmark=True)")
             torch.backends.cudnn.deterministic = False
             torch.backends.cudnn.benchmark = True


def setup_training_config(args) -> Union[TrainingConfig, DeCLIPLitDataTrainingConfig]:
    """Create training configuration from preset and command-line args"""
    default_monitor_metric = None
    default_monitor_mode = "min"
    current_config_class = TrainingConfig # Default base
    using_litdata = False 

    # Start with the selected preset config
    if args.config_preset == "declip_yfcc15m_litdata_vitb32":
        config = BASELINE_YFCC15M_VITB32_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
    elif args.config_preset == "declip_yfcc15m_litdata_resnet50":
        config = BASELINE_YFCC15M_RESNET50_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
    elif is_supervised_preset(args.config_preset):
        config, _ = load_supervised_training_config(args.config_preset)
        current_config_class = type(config)
    else:
        raise ValueError(f"Invalid config_preset: {args.config_preset}.")
    default_monitor_metric = config.monitor_metric
    default_monitor_mode = config.monitor_mode
    using_litdata = True

    config.wandb.enabled = bool(args.wandb)

    # Override with command-line args if they are provided
    # Data settings
    if args.dataset_type is not None:
        config.data.dataset_type = args.dataset_type
        # Update using_litdata flag based on final dataset_type
        using_litdata = (config.data.dataset_type == "yfcc15m_litdata")
    # HF settings
    if args.hf_dataset_name is not None: config.data.hf_dataset_name = args.hf_dataset_name
    if args.hf_cache_dir is not None: config.data.hf_cache_dir = args.hf_cache_dir
    if args.hf_streaming is not None: config.data.hf_streaming = args.hf_streaming
    # LitData settings
    if args.use_litdata is not None: # Explicit override
        config.data.use_litdata = args.use_litdata
        using_litdata = config.data.use_litdata
    else: # Infer from type if not explicitly set
         config.data.use_litdata = using_litdata

    if args.parquet_data_dir is not None: config.data.parquet_data_dir = args.parquet_data_dir

    if args.image_col is not None: config.data.image_col = args.image_col
    if args.text_col is not None: config.data.text_col = args.text_col
    if args.tokenizer_name is not None: config.data.tokenizer_name = args.tokenizer_name
    if args.batch_size is not None: config.data.batch_size = args.batch_size
    if args.num_workers is not None: config.data.num_workers = args.num_workers
    if args.val_dataset_root is not None: config.data.val_dataset_root = args.val_dataset_root
    if args.imagenet_train_dir is not None: config.data.data_dir = args.imagenet_train_dir
    if args.imagenet_val_dir is not None: config.data.val_data_dir = args.imagenet_val_dir
    if args.eval_batch_size is not None: config.data.eval_batch_size = args.eval_batch_size


    # Training settings
    if args.epochs is not None: config.epochs = args.epochs
    if args.seed is not None: config.seed = args.seed
    if args.accelerator is not None: config.accelerator = args.accelerator
    if args.devices is not None: config.devices = args.devices # Parsed later
    if args.gradient_clip_val is not None: config.gradient_clip_val = args.gradient_clip_val
    if args.accumulate_grad_batches is not None: config.accumulate_grad_batches = args.accumulate_grad_batches
    if args.log_every_n_steps is not None: config.log_every_n_steps = args.log_every_n_steps
    config.save_dir = str(args.save_dir or config.save_dir or paths.path("runs_root"))
    if args.save_top_k is not None: config.save_top_k = args.save_top_k
    if args.validate_every_n_steps is not None:
        config.validate_every_n_steps = args.validate_every_n_steps
        config.validate_every_n_epoch = None  # Ensure epoch-based validation is disabled
    elif args.validate_every_n_epoch is not None:
        config.validate_every_n_epoch = args.validate_every_n_epoch
        config.validate_every_n_steps = None
    
    config.save_every_n_train_steps = None

    # Optimizer settings
    if args.optimizer is not None: config.optimizer.name = args.optimizer
    if args.lr is not None: config.optimizer.lr = args.lr
    if args.weight_decay is not None: config.optimizer.weight_decay = args.weight_decay

    # Scheduler settings
    if args.scheduler is not None: config.scheduler.name = args.scheduler
    if args.warmup_steps is not None: config.scheduler.warmup_steps = args.warmup_steps
    config.scheduler.epochs = None # Not directly used by scheduler config
    # Ensure epochs are set
    if not config.epochs or config.epochs <= 0:
         rank_zero_info(f"Warning: Invalid epochs value ({config.epochs}). Defaulting to 1 epoch.")
         config.epochs = 1


    # Distributed training settings
    if args.strategy is not None: config.distributed.strategy = args.strategy
    if args.sync_batchnorm is not None: config.distributed.sync_batchnorm = args.sync_batchnorm # Handled later based on distribution status
    if args.find_unused_parameters is not None: config.distributed.find_unused_parameters = args.find_unused_parameters
    if args.num_nodes is not None: config.distributed.num_nodes = args.num_nodes

    # Wandb settings - only override if wandb is enabled
    if config.wandb.enabled:
        if args.wandb_project is not None: 
            config.wandb.project = args.wandb_project
        if args.experiment_name is not None: 
            config.wandb.name = args.experiment_name # Set later if None
        if args.wandb_entity is not None: 
            config.wandb.entity = args.wandb_entity
        if args.wandb_tags is not None: 
            config.wandb.tags = args.wandb_tags.split(",") if args.wandb_tags else []
        # Handle log_model boolean/string logic
        if args.wandb_log_model is not None:
            log_model_arg = args.wandb_log_model.lower()
            if log_model_arg == 'false':
                config.wandb.log_model = False
            elif log_model_arg in ['true', 'all', 'best']:
                config.wandb.log_model = log_model_arg # Keep 'all' or 'best' as string
            else:
                logger.warning(f"Invalid --wandb_log_model value: {args.wandb_log_model}. Using preset default: {config.wandb.log_model}.")
        if args.wandb_save_code is not None: 
            config.wandb.save_code = args.wandb_save_code
    else:
        rank_zero_info("W&B is disabled - ignoring all wandb-related CLI arguments.")

    if args.freeze_vision_blocks is not None:
        config.freeze_vision_blocks = args.freeze_vision_blocks

    if args.perturb_type is not None:
        config.perturb_type = args.perturb_type
    if args.perturb_epoch is not None:
        config.perturb_epoch = args.perturb_epoch
    if args.perturb_length is not None:
        config.perturb_length = args.perturb_length
    if args.perturb_seed is not None:
        config.perturb_seed = args.perturb_seed


    # Set monitor metric/mode after the overrides above so they take effect.
    config.monitor_metric = args.monitor_metric if args.monitor_metric is not None else default_monitor_metric
    config.monitor_mode = args.monitor_mode if args.monitor_mode is not None else default_monitor_mode

    if args.test:
        rank_zero_info("TEST MODE ENABLED: Overriding configuration for quick pipeline verification.")
        config.epochs = 1
        config.log_every_n_steps = 1
        config.validate_every_n_epoch = 1
        config.validate_every_n_steps = None
        config.save_top_k = 1
        # Add test mode tag to wandb
        if hasattr(config, 'wandb') and config.wandb.tags:
            config.wandb.tags.append("test-mode")
        rank_zero_info("Test mode: Training for 1 epoch with 5 training steps and frequent logging.")

    # Make sure the final config object is of the correct Pydantic type
    # This is important if args heavily modified a base TrainingConfig
    if not isinstance(config, current_config_class) and current_config_class != TrainingConfig:
        try:
            config_dict = config.dict()
            # Filter dict for fields present in target class to avoid errors
            valid_fields = {f for f in current_config_class.model_fields.keys()}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
            # Special handling for nested Pydantic models if direct casting fails
            # For now, assume direct casting from filtered dict works for major structures
            config = current_config_class(**filtered_dict)
            rank_zero_info(f"Config re-casted to {current_config_class.__name__} after applying CLI overrides.")
        except Exception as e:
            rank_zero_info(f"Warning: Could not re-cast config to {current_config_class.__name__} after overrides: {e}. Using original type {type(config).__name__}.")

    rank_zero_info(f"Final Configuration Type: {type(config).__name__}")
    rank_zero_info(f"Training for {config.epochs} epochs.")
    rank_zero_info(f"Using LitData: {config.data.use_litdata}")
    rank_zero_info(f"Monitoring metric: '{config.monitor_metric}' (mode: {config.monitor_mode})")
    if config.monitor_metric is None and (args.save_top_k is None or config.save_top_k > 0):
        rank_zero_info("No monitor metric set. Checkpointing will only save the 'last' checkpoint.")

    return config

class RestoreEpochCallback(Callback):
    """Restores trainer.current_epoch (and global_step) when resuming from a checkpoint.

    CLIPLightningModule.on_load_checkpoint removes the 'loops' key from the
    checkpoint dict so that LitData's StreamingDataset does not re-validate its
    input_dir / num_workers against the original run's cache paths.  The side
    effect is that Lightning cannot restore the epoch counter from the loop state
    dict, causing trainer.current_epoch to reset to 0.

    Fires in on_fit_start (before train_dataloader / LitData StreamingDataset is
    constructed) so the LitData shuffle/chunk order is derived from the correct
    resumed epoch from the very first batch — not from epoch=0 with a later
    set_epoch() catch-up that may not invalidate already-prefetched chunks.
    """

    def __init__(self, resume_epoch: int, resume_global_step: int):
        self._epoch = resume_epoch          # top-level "epoch" value from the checkpoint
        self._global_step = resume_global_step

    def _apply(self, trainer: "pl.Trainer", phase: str) -> None:
        if self._epoch >= 0:
            # The checkpoint was saved at the END of self._epoch, so the next
            # epoch to run is self._epoch + 1.
            #
            # Lightning's `FitLoop.done` reads `epoch_progress.current.processed`
            # to decide when to stop (NOT `completed` — that's only updated after
            # post-epoch callbacks). Setting just `completed` leaves `processed`
            # at 0, so Lightning trains max_epochs MORE epochs on every resume.
            # Set every Progress counter (ready/started/processed/completed) on
            # both `current` and `total` so the stop-check fires correctly.
            next_epoch = self._epoch + 1
            for prog in (trainer.fit_loop.epoch_progress.current,
                         trainer.fit_loop.epoch_progress.total):
                prog.ready = next_epoch
                prog.started = next_epoch
                prog.processed = next_epoch
                prog.completed = next_epoch
            rank_zero_info(
                f"RestoreEpochCallback[{phase}]: trainer.current_epoch set to {next_epoch} "
                f"(checkpoint epoch={self._epoch}); processed/completed counters synced"
            )
        if self._global_step >= 0:
            try:
                epoch_loop = trainer.fit_loop.epoch_loop
                epoch_loop.batch_progress.total.completed = self._global_step
                epoch_loop.batch_progress.current.completed = self._global_step
                # trainer.global_step reads the optimizer-step counter, and every logger
                # (self.log, LearningRateMonitor) uses _batches_that_stepped as the step.
                # Both live in the dropped loop state, so without this a resumed run's
                # global_step and TensorBoard step axis restart at 0. The per-epoch reset
                # only clears the `current` counters, so these survive into training.
                step_progress = epoch_loop.automatic_optimization.optim_progress.optimizer.step.total
                step_progress.ready = self._global_step
                step_progress.completed = self._global_step
                epoch_loop._batches_that_stepped = self._global_step
                rank_zero_info(f"RestoreEpochCallback[{phase}]: global_step set to {self._global_step}")
            except AttributeError as e:
                rank_zero_info(f"RestoreEpochCallback[{phase}]: could not restore global_step ({e})")

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # First application: ensures sanity check runs at the resumed epoch.
        self._apply(trainer, phase="on_fit_start")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Re-apply: PL re-initialises fit_loop.epoch_progress.current between
        # sanity-check-end and first training epoch, which resets our values
        # to 0. on_train_start fires after sanity check and before the first
        # training epoch, so re-injecting here keeps trainer.current_epoch at
        # the resumed value for the actual training loop.
        self._apply(trainer, phase="on_train_start")


class StopAfterEpochCallback(Callback):
    """Stops training after the target epoch completes."""
    def __init__(self, stop_epoch: int):
        super().__init__()
        self.stop_epoch = stop_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.stop_epoch:
            rank_zero_info(f"[StopAfterEpoch] Stopping after epoch {trainer.current_epoch}")
            trainer.should_stop = True


class SubEpochCheckpointCallback(Callback):
    """Saves checkpoints at specific training step indices within a target epoch."""
    def __init__(self, target_epoch: int, save_steps: List[int], dirpath: str):
        super().__init__()
        self.target_epoch = target_epoch
        self.save_steps = set(save_steps)
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch != self.target_epoch:
            return
        if batch_idx not in self.save_steps:
            return
        filename = f"epoch_{self.target_epoch:03d}_step_{batch_idx:05d}.ckpt"
        path = os.path.join(self.dirpath, filename)
        trainer.save_checkpoint(path)
        rank_zero_info(f"[SubEpochCheckpoint] Saved {path} (batch_idx={batch_idx})")


class ResetModelCheckpointStateCallback(Callback):
    """Reinitializes ModelCheckpoint bookkeeping after resume so all ranks agree.

    When resuming from a checkpoint with a different `dirpath` than the original
    run, PL's ModelCheckpoint reload only restores `best_model_path` (which
    points at a file in the OLD dirpath that doesn't exist in the new run) and
    leaves `best_model_score`, `best_k_models`, `kth_best_model_path`, and
    `kth_value` at their __init__ defaults.  This produced a DDP deadlock at the
    validation→training transition where rank 0 was 2 collectives ahead of
    ranks 1-7 on a NumelIn=1 ALLREDUCE — i.e. ModelCheckpoint's save/skip
    decision diverged across ranks.

    Fires in on_fit_start (after PL restore_modules_and_callbacks, before any
    validation or training) and resets every ModelCheckpoint instance so the
    subsequent first save decision is identical on every rank.
    """

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for cb in trainer.callbacks:
            if not isinstance(cb, ModelCheckpoint):
                continue
            cb.best_model_path = ""
            cb.best_model_score = None
            cb.best_k_models = {}
            cb.kth_best_model_path = ""
            cb.kth_value = float("-inf") if cb.mode == "max" else float("inf")
            cb.last_model_path = ""
            rank_zero_info(
                f"ResetModelCheckpointStateCallback: cleared {cb.__class__.__name__} "
                f"(dirpath={cb.dirpath}, mode={cb.mode})"
            )


def setup_callbacks(config: TrainingConfig, experiment_name: str, args):
    callbacks = []
    ckpt_path = os.path.join(config.save_dir, experiment_name, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    rank_zero_info(f"Checkpoints will be saved to: {ckpt_path}")

    monitor_enabled = config.monitor_metric is not None

    # Callback: Save based on monitor metric (top-k) AND save the last checkpoint
    if monitor_enabled:
        monitor_key_slug = config.monitor_metric.replace('/', '_')
        monitor_filename_format = f"monitor_{monitor_key_slug}" + "-epoch_{epoch:03d}"
        top_k_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path,
            filename=monitor_filename_format,
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            save_top_k=config.save_top_k,
            save_last=True, # Ensure the last checkpoint is always saved
        )
        callbacks.append(top_k_checkpoint_callback)
        rank_zero_info(f"Checkpointing Strategy: Monitor='{config.monitor_metric}', Mode='{config.monitor_mode}', Save Top K={config.save_top_k}, Save Last=True")
    else:
         # Fallback: If no monitoring is enabled, just save the last checkpoint
         last_checkpoint_callback = ModelCheckpoint(
             dirpath=ckpt_path,
             filename="last-epoch_{epoch:03d}",
             save_last=True,
             save_top_k=0 # Explicitly disable top-k
         )
         callbacks.append(last_checkpoint_callback)
         rank_zero_info("Checkpointing Strategy: No monitor metric set. Saving only the last checkpoint.")

    lr_monitor = LearningRateMonitor(logging_interval="step") # Log LR every step
    callbacks.append(lr_monitor)

    if args.early_stopping and monitor_enabled:
        early_stop_callback = EarlyStopping(
            monitor=config.monitor_metric,
            min_delta=1e-5,
            patience=args.patience,
            verbose=True,
            mode=config.monitor_mode,
            check_on_train_epoch_end=False # Check after validation epoch
        )
        callbacks.append(early_stop_callback)
        rank_zero_info(f"Early Stopping Enabled: Monitor='{config.monitor_metric}', Patience={args.patience}")
    elif args.early_stopping and not monitor_enabled:
         rank_zero_info("Early stopping requested but no monitor_metric set. Disabling early stopping.")

    if getattr(args, 'save_every_epoch', False):
        every_epoch_dir = os.path.join(ckpt_path, "all_epochs")
        os.makedirs(every_epoch_dir, exist_ok=True)
        every_epoch_callback = ModelCheckpoint(
            dirpath=every_epoch_dir,
            filename="epoch_{epoch:03d}",
            # Lightning defaults auto_insert_metric_name=True, which would turn
            # filename="epoch_{epoch:03d}" into "epoch_epoch=015.ckpt" (the substituted
            # variable gets "epoch=" prepended). Disable so we get the intended
            # "epoch_015.ckpt" naming the fork-at-onset runs resume from.
            auto_insert_metric_name=False,
            save_top_k=-1,
            every_n_epochs=1,
            monitor=None,
            save_on_train_epoch_end=True,
        )
        callbacks.append(every_epoch_callback)
        rank_zero_info(f"Save-every-epoch enabled: checkpoints at {every_epoch_dir}/epoch_NNN.ckpt")

    if getattr(args, 'stop_epoch', None) is not None:
        callbacks.append(StopAfterEpochCallback(args.stop_epoch))
        rank_zero_info(f"Will stop training after epoch {args.stop_epoch}")

    if getattr(args, 'sub_epoch_ckpt_steps', None):
        steps = [int(s) for s in args.sub_epoch_ckpt_steps.split(',')]
        target_epoch = args.sub_epoch_ckpt_epoch
        if target_epoch is None:
            perturb_epoch = getattr(config, 'perturb_epoch', None)
            perturb_length = getattr(config, 'perturb_length', 1) or 1
            if perturb_epoch is not None:
                target_epoch = perturb_epoch + perturb_length
            else:
                raise ValueError("--sub_epoch_ckpt_steps requires --sub_epoch_ckpt_epoch (or an active perturbation to infer it from)")
        sub_epoch_dir = os.path.join(ckpt_path, "sub_epoch")
        callbacks.append(SubEpochCheckpointCallback(target_epoch, steps, sub_epoch_dir))
        rank_zero_info(f"Sub-epoch checkpointing: epoch {target_epoch}, steps {sorted(steps)}, dir {sub_epoch_dir}")

    return callbacks

def setup_loggers(config: TrainingConfig, experiment_name: str, args):
    loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=config.save_dir,
        name=experiment_name,
    )
    loggers.append(tb_logger)

    # Only setup wandb if it's enabled in config
    if config.wandb.enabled:
        is_torchrun, env_info = detect_torchrun_environment()
        # Only init wandb on global rank 0
        rank = int(env_info.get('RANK', 0)) if is_torchrun else 0
        if rank == 0:
            try:
                # Use experiment name for wandb run name if not overridden
                wandb_name = config.wandb.name if config.wandb.name else experiment_name
                wandb_logger = WandbLogger(
                    project=config.wandb.project,
                    name=wandb_name,
                    entity=config.wandb.entity,
                    tags=config.wandb.tags,
                    log_model=config.wandb.log_model, # Pass the processed value
                    save_code=config.wandb.save_code,
                    # Group runs by node if using torchrun across nodes
                    group=f"exp_{experiment_name}_node{env_info.get('NODE_RANK', 0)}" if is_torchrun and config.distributed.num_nodes > 1 else f"exp_{experiment_name}",
                    job_type="training",
                    # Pass config dict to wandb
                    config=config.dict() # Log the final merged config
                )
                loggers.append(wandb_logger)
                logger.info(f"Weights & Biases logging enabled for project '{config.wandb.project}', run '{wandb_name}'.")
                logger.info(f"W&B tags: {config.wandb.tags}")
            except Exception as e:
                logger.error(f"Failed to initialize WandbLogger: {e}. Disabling wandb.")
                config.wandb.enabled = False # Disable it in the config object too
        else:
            logger.info(f"Skipping wandb initialization on rank {rank}.")
            # Ensure wandb is marked disabled for non-zero ranks if logic depends on it later
            config.wandb.enabled = False
    else:
        logger.info("W&B logging is disabled.")

    return loggers

def main():
    args = parse_args()
    is_torchrun, env_info = detect_torchrun_environment()
    rank = int(env_info.get('RANK', 0)) if is_torchrun else 0

    # Setup configuration based on preset and overrides
    config = setup_training_config(args)
    
    # Load the appropriate model config for the selected training config.
    # Supervised presets use a SupervisedModelConfig (timm arch + head); CLIP uses CLIPConfig.
    if is_supervised_preset(args.config_preset):
        _, model_config = load_supervised_training_config(args.config_preset)
    else:
        _, model_config = get_config(args.config_preset)

    # Generate experiment name if needed
    experiment_name = args.experiment_name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Make name more descriptive
        preset_name = args.config_preset or config.data.dataset_type
        experiment_name = f"{preset_name}_{timestamp}"
        if is_torchrun:
            experiment_name += f"_ddp_node{env_info.get('NODE_RANK', 0)}" # Add node rank

    # Set seed *before* initializing models or dataloaders
    # Use the final seed value from the config object
    set_seed(config.seed, getattr(config, "deterministic", True))
    rank_zero_info(f"Set seed to {config.seed} (Deterministic: {getattr(config, 'deterministic', True)})")

    if torch.cuda.is_available():
        try:
            # Set to 'medium' for good balance of performance and precision on A100 GPUs
            torch.set_float32_matmul_precision('medium')
            rank_zero_info("Set torch.set_float32_matmul_precision to 'medium' for Tensor Cores optimization")
        except Exception as e:
            rank_zero_info(f"Failed to set torch matmul precision: {e}")

    num_avail_gpus = count_available_gpus()
    is_distributed = False
    world_size = 1
    local_batch_size = config.data.batch_size # Batch size per device

    if isinstance(config.devices, str) and config.devices != "auto":
        try:
            if "," in config.devices:
                devices_list = [int(x.strip()) for x in config.devices.split(",")]
                config.devices = devices_list
                num_devices_per_node = len(devices_list)
                world_size = num_devices_per_node * config.distributed.num_nodes
                is_distributed = world_size > 1
                rank_zero_info(f"Using specified devices: {devices_list} (Count: {num_devices_per_node})")
            else:
                num_devices_per_node = int(config.devices)
                config.devices = num_devices_per_node # Store the count
                world_size = num_devices_per_node * config.distributed.num_nodes
                is_distributed = world_size > 1
                rank_zero_info(f"Using specified device count: {num_devices_per_node}")
        except ValueError:
            rank_zero_info(f"Invalid devices format: {config.devices}. Using 'auto'.")
            config.devices = "auto"

    if config.devices == "auto":
        if is_torchrun:
             # Read from torchrun env vars
             local_world_size = int(env_info.get('LOCAL_WORLD_SIZE', 1))
             num_nodes = int(env_info.get('NNODES', config.distributed.num_nodes)) # Use NNODES from torchrun if available
             world_size = int(env_info.get('WORLD_SIZE', 1))

             if num_nodes != config.distributed.num_nodes:
                  rank_zero_info(f"Overriding config.distributed.num_nodes ({config.distributed.num_nodes}) with torchrun NNODES ({num_nodes})")
                  config.distributed.num_nodes = num_nodes

             config.devices = local_world_size # Let Lightning handle device IDs per node based on count
             is_distributed = world_size > 1
             rank_zero_info(f"torchrun detected. world_size={world_size}, devices_per_node={local_world_size}, num_nodes={num_nodes}")
        elif num_avail_gpus > 0:
             # Use all available GPUs on the node if not using torchrun
             config.devices = num_avail_gpus # Use the count
             world_size = num_avail_gpus * config.distributed.num_nodes # Assumes same num GPUs per node
             is_distributed = world_size > 1
             rank_zero_info(f"Auto-detected {num_avail_gpus} GPUs per node. world_size={world_size}")
        else:
              # CPU training
             config.devices = 1
             world_size = 1 * config.distributed.num_nodes
             is_distributed = world_size > 1
             rank_zero_info("Using CPU.")

    # Final check on distribution status
    if world_size <= 1:
        is_distributed = False
        if is_torchrun:
             rank_zero_info("torchrun detected but world_size <= 1. Running in non-distributed mode.")


    # Set default sync_batchnorm if distributed
    if is_distributed and config.distributed.sync_batchnorm is None:
         config.distributed.sync_batchnorm = True
         rank_zero_info("Distributed training detected. Enabling sync_batchnorm by default.")
    elif config.distributed.sync_batchnorm is None:
         config.distributed.sync_batchnorm = False # Default to False if not distributed

    max_epochs_trainer = config.epochs if config.epochs and config.epochs > 0 else -1
    max_steps_trainer = -1 # Always epoch-based now
    if max_epochs_trainer == -1:
        rank_zero_info("Warning: Trainer max_epochs is not set. Training will rely on dataset length or manual stop.")
    else:
        rank_zero_info(f"Trainer will run for max_epochs = {max_epochs_trainer}")

    # When resuming, --epochs is the *total* epoch count across all runs.
    # max_epochs_trainer is kept unchanged; the epoch counter is restored by
    # RestoreEpochCallback (see below) because on_load_checkpoint intentionally
    # drops the 'loops' key to prevent LitData from re-validating stale paths.
    resume_epoch: int = -1
    resume_global_step: int = -1
    if args.resume_ckpt_path:
        try:
            ckpt_meta = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
            resume_epoch = int(ckpt_meta.get("epoch", -1))
            resume_global_step = int(ckpt_meta.get("global_step", -1))
            rank_zero_info(
                f"Resume: checkpoint={args.resume_ckpt_path} "
                f"(epoch={resume_epoch}, global_step={resume_global_step}). "
                f"max_epochs kept at {max_epochs_trainer}. "
                f"Epoch counter will be restored by RestoreEpochCallback."
            )
        except Exception as e:
            rank_zero_info(
                f"Resume: could not read epoch/step from checkpoint ({e}). "
                f"max_epochs kept at {max_epochs_trainer}."
            )


    val_check_interval_trainer = None
    check_val_every_n_epoch_trainer = None

    # If validate_every_n_steps was provided (not None), use it and disable epoch checking.
    if config.validate_every_n_steps is not None:
        val_check_interval_trainer = config.validate_every_n_steps
        if isinstance(val_check_interval_trainer, float):
            # Float means fraction-of-epoch; PL requires check_val_every_n_epoch to be set (not None)
            check_val_every_n_epoch_trainer = 1
            rank_zero_info(f"Validation check mode: Every {val_check_interval_trainer:.2f} fraction of epoch.")
        else:
            check_val_every_n_epoch_trainer = None  # Step-based: disable epoch check
            rank_zero_info(f"Validation check mode: Every {val_check_interval_trainer} steps.")
    # Otherwise, use epoch-based checking (defaulting to 1 if config value is None)
    else:
        check_val_every_n_epoch_trainer = config.validate_every_n_epoch if config.validate_every_n_epoch else 1
        val_check_interval_trainer = None  # Disable step check
        rank_zero_info(f"Validation check mode: Every {check_val_every_n_epoch_trainer} epoch(s).")


    # Setup loggers and callbacks
    loggers = setup_loggers(config, experiment_name, args)
    if config.wandb.enabled and config.wandb.name is None: config.wandb.name = experiment_name
    callbacks = setup_callbacks(config, experiment_name, args)
    if resume_epoch >= 0:
        # Reset ModelCheckpoint bookkeeping FIRST on resume so subsequent
        # callback hooks (including RestoreEpochCallback below and any
        # validation→save decisions) see clean, all-ranks-identical state.
        callbacks.insert(0, ResetModelCheckpointStateCallback())
        rank_zero_info("ResetModelCheckpointStateCallback registered (clears stale ModelCheckpoint state on resume)")
        callbacks.append(RestoreEpochCallback(resume_epoch, resume_global_step))
        rank_zero_info(
            f"RestoreEpochCallback registered: current_epoch → {resume_epoch + 1}, "
            f"global_step → {resume_global_step}"
        )

    # Wrap dataloader creation in a LightningDataModule so setup() runs post-DDP-init.
    # This fixes the "torch.distributed not initialized when creating val_sampler" warning
    # that caused every rank to iterate the full CIFAR val sets (4x redundant eval).
    rank_zero_info(f"Creating dataloaders for dataset type: {config.data.dataset_type}")
    rank_zero_info(f"Multi-validation enabled: {args.multi_val}")
    task = getattr(config, "task", "clip")
    if task == "supervised":
        datamodule = ImageNetDataModule(
            config=config.data,
            seed=config.seed,
            distributed=is_distributed,
        )
    else:
        datamodule = YFCC15MDataModule(
            config=config.data,
            seed=config.seed,
            distributed=is_distributed,
            multi_val=args.multi_val,
        )

    # The CLIP arm trains on a LitData streaming set, whose length Lightning can't
    # report to configure_optimizers() in time (num_training_batches == inf). Without
    # a size, the cosine horizon collapses to a max(10000, warmup*5)*epochs fallback
    # (~400k steps vs the real ~9.7k → the LR never anneals). Read the index length
    # here (cheap index.json read) so the module sizes the schedule as dataset/global_batch.
    # Supervised keeps 0 — SupervisedLightningModule derives steps from a finite dataset.
    train_dataset_size_for_module = 0
    if task != "supervised":
        from temporal_perturbations.scratch_training.dataset import yfcc15m_train_size
        train_dataset_size_for_module = yfcc15m_train_size(config.data)
        if train_dataset_size_for_module > 0:
            rank_zero_info(f"YFCC15M train size = {train_dataset_size_for_module} (sizes the cosine LR horizon).")
        else:
            rank_zero_info("Could not read YFCC15M train size; LR schedule falls back to the "
                           "num_training_batches heuristic (schedule may not anneal).")

    has_val = config.data.dataset_type in ('yfcc15m_litdata', 'imagenet1k')
    if not has_val and config.monitor_metric and 'val/' in config.monitor_metric:
        rank_zero_info(f"Monitoring validation metric '{config.monitor_metric}' but no validation dataloader is expected. Checkpointing/EarlyStopping might fail.")


    # Create model — branch on task. CLIPLightningModule uses CLIPConfig (vision+text encoders);
    # SupervisedLightningModule uses SupervisedModelConfig (timm classifier).
    if task == "supervised":
        rank_zero_info(f"Initializing SupervisedLightningModule (arch={model_config.name}, num_classes={model_config.num_classes})")
        model = SupervisedLightningModule(
            model_config=model_config,
            training_config=config,
            train_dataset_size=train_dataset_size_for_module,
        )
    else:
        rank_zero_info(f"Initializing CLIPLightningModule for training mode: {config.data.dataset_type}")
        rank_zero_info(f"Using model config: {model_config.vision_cfg.layers} layers, {model_config.embed_dim} embed_dim, patch_size {model_config.vision_cfg.patch_size}")
        model = CLIPLightningModule(
            model_config=model_config,
            training_config=config,
            train_dataset_size=train_dataset_size_for_module,
        )

    if args.compile:
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version >= (2, 0):
            rank_zero_info("Attempting to compile the model with torch.compile...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                rank_zero_info("Model compiled successfully.")
            except Exception as e:
                rank_zero_info(f"torch.compile failed: {e}. Proceeding without compilation.")
        else:
            rank_zero_info(f"torch.compile requires PyTorch >= 2.0, but found {torch.__version__}. Skipping compilation.")


    # Configure training strategy for Lightning Trainer
    trainer_strategy = "auto"
    if is_distributed:
        # Use strategy from config if specified
        if config.distributed.strategy and config.distributed.strategy.lower() != "ddp":
             trainer_strategy = config.distributed.strategy
             rank_zero_info(f"Using specified strategy: {trainer_strategy}")
        else:
             trainer_strategy = DefaultStreamDDPStrategy(
                find_unused_parameters=config.distributed.find_unused_parameters,
                gradient_as_bucket_view=True,
             )
             rank_zero_info(f"Using DDP strategy (gradient_as_bucket_view=True, find_unused_parameters={config.distributed.find_unused_parameters})")

    # calculate accumulate_grad_batches
    if not config.accumulate_grad_batches:
        accumulate_grad_batches = config.target_batch_size // config.data.batch_size // world_size
        rank_zero_info(f"Calculated accumulate_grad_batches: {accumulate_grad_batches}")
    else:
        accumulate_grad_batches = config.accumulate_grad_batches
        rank_zero_info(f"Using specified accumulate_grad_batches: {accumulate_grad_batches}")
    
    # Create trainer
    rank_zero_info(f"Configuring Trainer with max_epochs={max_epochs_trainer}, max_steps={max_steps_trainer}")
    
    # Test mode configuration
    limit_train_batches = 100 if args.test else 1.0
    if args.test:
        rank_zero_info(f"Test mode: Limiting training to {limit_train_batches} batches per epoch.")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs_trainer,
        max_steps=max_steps_trainer, # Should be -1
        limit_train_batches=limit_train_batches,  # Limit to 5 batches in test mode
        enable_progress_bar=(rank == 0), # Only show progress on rank 0
        accelerator=config.accelerator if not args.test else "cpu",
        devices=config.devices if not args.test else 1,
        num_nodes=config.distributed.num_nodes if not args.test else 1,
        strategy=trainer_strategy,
        precision=config.precision,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=val_check_interval_trainer,
        check_val_every_n_epoch=check_val_every_n_epoch_trainer,
        num_sanity_val_steps=-1 if has_val else 0, # Full sanity check if val exists, else 0
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=config.deterministic,
        benchmark=not config.deterministic,
        sync_batchnorm=config.distributed.sync_batchnorm,
    )

    # Add hook for setting epoch in distributed samplers/datasets
    # Handles both a DistributedSampler and a LitData StreamingDataset.
    class SetEpochHook(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            current_epoch = trainer.current_epoch
            train_loader = trainer.train_dataloader
            if train_loader is None: return

            # Handle standard DataLoader with DistributedSampler
            sampler = getattr(train_loader, 'sampler', None)
            # Handle nested sampler if DataLoader wraps another
            if hasattr(sampler, 'sampler') and isinstance(sampler.sampler, torch.utils.data.DistributedSampler):
                 sampler.sampler.set_epoch(current_epoch)
            elif isinstance(sampler, torch.utils.data.DistributedSampler):
                 sampler.set_epoch(current_epoch)

            # Handle LitData StreamingDataLoader/StreamingDataset
            dataset = getattr(train_loader, 'dataset', None)
            if hasattr(dataset, 'set_epoch') and callable(dataset.set_epoch):
                 try:
                     dataset.set_epoch(current_epoch)
                 except Exception as e:
                      rank_zero_info(f"Warning: Failed to set epoch on LitData dataset: {e}")


    trainer.callbacks.append(SetEpochHook())


    rank_zero_info("Starting training...")
    if not has_val and (val_check_interval_trainer is not None or check_val_every_n_epoch_trainer is not None):
        rank_zero_info("Validation is configured to run, but no validation dataloader was created. Validation steps will be skipped.")

    resume_path = args.resume_ckpt_path if args.resume_ckpt_path else None
    if resume_path:
        rank_zero_info(f"Resuming training from checkpoint: {resume_path}")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resume_path
    )

    # YFCC15M test_step is a no-op (logs 0.0); skipping trainer.test() saves a
    # model reload + DDP sync.


    rank_zero_info("Training finished.")
    # PTL handles process group cleanup

    return 0


if __name__ == "__main__":
    main() 
