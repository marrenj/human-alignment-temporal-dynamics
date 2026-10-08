"""Train CLIP-HBA with DoRA on behavioral similarity targets, with controlled perturbation windows and checkpoint resumption."""

import os
import sys
import random
import csv
import json
import yaml
import wandb
from tqdm import tqdm
from numpy.random import set_state as np_set_state
import torch 
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, IterableDataset, random_split, Subset
from torch.nn import DataParallel
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from temporal_perturbations.training.test_loop import evaluate_model
from temporal_perturbations.utils.save_random_states import save_random_states
from temporal_perturbations.models.clip_hba.clip_hba_utils import save_dora_parameters
from temporal_perturbations.models.factory import build_model
from temporal_perturbations.utils.seed import seed_everything
from temporal_perturbations.utils.logging import setup_logger
from temporal_perturbations.utils.count_parameters import count_trainable_parameters
from temporal_perturbations.models.clip_hba.clip_hba_utils import load_dora_checkpoint
from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset
from temporal_perturbations.data.imagenet_dataset import ImagenetDataset
from temporal_perturbations.perturbations.perturbation_utils import choose_perturbation_strategy


def build_optimizer(model, config):
    """Build an SGD or AdamW optimizer from ``config['opt']``."""
    opt_name    = str(config.get('opt')).lower()
    lr          = float(config['lr'])
    weight_decay = float(config.get('weight_decay', 0.0))

    if opt_name == 'sgd':
        momentum = float(config.get('momentum', 0.9))
        return SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
        )
    elif opt_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(
            f"Optimizer '{opt_name}' not supported. Use 'sgd' or 'adamw'."
        )


def build_scheduler(optimizer, config, epochs: int):
    """Build the LR scheduler named by ``config['lr_scheduler']`` (cosine, cosine with linear warmup, or none)."""
    name = str(config.get('lr_scheduler', 'none')).lower().strip()
    if not name or name == 'none':
        return None

    warmup_epochs = int(config.get('lr_warmup_duration'))


    if name == 'cosineannealinglrwithwarmup':
        cosine_epochs = max(1, epochs - warmup_epochs)
        if warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer,
                start_factor=1e-6,   # near-zero start
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=0
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    if name == 'cosineannealinglr':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    raise ValueError(
        f"LR scheduler '{name}' not supported. "
        "Use 'cosineannealinglrwithwarmup', 'cosineannealinglr', or 'none'."
    )


def get_wandb_tags(config):
    """Return wandb tags describing the run configuration."""
    training_mode = config.get('training_mode')

    # Use clip_hba_backbone for finetune, architecture for scratch
    backbone_tag = (
        config.get('clip_hba_backbone')
        if training_mode == 'finetune'
        else config.get('architecture')
    )

    tags = [
        f"training_mode_{training_mode}",
        f"backbone_{backbone_tag}",
        f"perturb_type_{config['perturb_type']}",
        f"dataset_type_{config['dataset_type']}",
        f"init_seed_{config['random_seed']}",
    ]

    if training_mode == 'finetune':
        tags.append(f"rank_{config.get('rank')}")

    if config.get('behavioral_rsa'):
        tags.append("behavioral_rsa")
    
    perturb_type = str(config.get('perturb_type')).lower()
    if perturb_type != 'none':
        tags.append(f"perturb_epoch_{config.get('perturb_epoch')}")
        tags.append(f"perturb_length_{config.get('perturb_length')}")
        tags.append(f"perturb_seed_{config.get('perturb_seed')}")
    
    if config.get('wandb_tags'):
        tags.extend(config.get('wandb_tags'))
    
    return tags


def get_run_name(config):
    """
    Build a deterministic run name shared by wandb and local save paths.'

    Works for both training modes (scratch and finetune).
    """
    def _sanitize(name: str) -> str:
        # Make a filesystem-safe, lowercase token (replace slashes/spaces)
        return name.replace("-", "_").replace("/", "_").replace(" ", "_").lower()

    if config.get('wandb_run_name'):
        return config['wandb_run_name']
    
    training_mode = config.get('training_mode')

    if training_mode == 'finetune':
        backbone_token = _sanitize(config.get('clip_hba_backbone'))
        rank_token = f"rank{config.get('rank')}"
    else:
        backbone_token = _sanitize(config.get('architecture'))
        rank_token = "scratch"

    perturb_type = str(config.get('perturb_type', 'none'))
    normalized_ptype = perturb_type.lower()

    if normalized_ptype == 'none':
        return (
            f"{backbone_token}_"
            f"{rank_token}_"
            f"perturb-type-none_"
            f"init-seed{config.get('random_seed')}"
            f"behavioral-rsa-{config.get('behavioral_rsa')}"
        )

    return (
        f"{backbone_token}_"
        f"{rank_token}_"
        f"perturb-type-{perturb_type}_"
        f"epoch{config.get('perturb_epoch')}_"
        f"length{config.get('perturb_length')}_"
        f"perturb-seed{config.get('perturb_seed')}_"
        f"init-seed{config.get('random_seed')}"
        f"behavioral-rsa-{config.get('behavioral_rsa')}"
    )


def init_wandb(config, resume_epoch=0):
    """
    Initialize Weights & Biases run with proper configuration.
    
    Args:
        config: Configuration dictionary
        resume_epoch: Epoch to resume from
        
    Returns:
        wandb.Run: Weights & Biases run
    """
    resume_mode = "allow" if resume_epoch > 0 else None
    run_id = config.get('wandb_run_id', None)

    # If project/entity are missing or blank, run wandb in offline mode
    project = config.get('wandb_project')
    entity = config.get('wandb_entity')
    offline_mode = (not project) or (entity is None or entity == "")
    if offline_mode:
        if os.environ.get("WANDB_MODE") != "disabled":
            os.environ["WANDB_MODE"] = "offline"
        project = project or "offline-run"

    # Create descriptive run name (shared with filesystem path)
    run_name = get_run_name(config)
    
    run = wandb.init(
        project=project,
        entity=entity if not offline_mode else None,
        name=run_name,
        id=run_id,
        resume=resume_mode,
        config=config,
        tags=get_wandb_tags(config),
        notes=config.get('wandb_notes', ''),
        save_code=True,
    )
    
    config['wandb_run_id'] = run.id
    config['wandb_run_name'] = run.name or run_name
    return run


def log_model_architecture(model, logger):
    """
    Log model architecture details to wandb.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if wandb.run is not None:
        wandb.config.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
        })
        trainable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
        wandb.config.update({'trainable_layers': trainable_layers})
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")


def _resuming_into_save_path(config) -> bool:
    """True when explicit checkpoint resume targets the same directory as save_path."""
    resume_path = config.get("resume_checkpoint_path")
    if not resume_path or config.get("resume_from_epoch") is None:
        return False
    save_path = config.get("save_path")
    if not save_path:
        return False
    return os.path.realpath(os.path.normpath(str(resume_path))) == os.path.realpath(
        os.path.normpath(str(save_path))
    )


def setup_paths(config):
    """
    Create and return all necessary directory paths for training outputs.
    
    Args:
        save_path: Root directory for saving all training artifacts
        
    Returns:
        tuple: (save_path, training_results_save_path, random_state_save_path)
    """
    os.makedirs(config['save_path'], exist_ok=True)
    training_results_save_path = os.path.join(config['save_path'], 'training_res.csv')
    random_state_save_path = os.path.join(config['save_path'], 'random_states')
    checkpoints_save_path = os.path.join(config['save_path'], 'model_checkpoints')

    skip_overwrite_prompt = (not sys.stdin.isatty()) or _resuming_into_save_path(config)

    # Check if files/directories already exist and ask for overwrite permission
    for path, description in [
        (training_results_save_path, 'training results file'),
        (random_state_save_path, 'random state save directory'),
        (checkpoints_save_path, 'model checkpoints save directory'),
    ]:
        if os.path.exists(path):
            if skip_overwrite_prompt:
                continue
            while True:
                response = input(f"{description.capitalize()} '{path}' already exists, are you sure you want to rewrite it? (Yes/No): ")
                if response.lower() in ["yes", "y"]:
                    break
                elif response.lower() in ["no", "n"]:
                    raise FileExistsError(f"Aborted because {description} '{path}' already exists.")
                else:
                    print("Please answer 'Yes' or 'No'.")
    return config['save_path'], training_results_save_path, random_state_save_path


def setup_dataset(config, logger):
    """Build train/val datasets. ImageNet (scratch) ships canonical train/val dirs, so only finetune does a random split.

    Returns (train_dataset, val_dataset, split_info, global_target_stats, inference_dataset).
    """
    training_mode = config.get('training_mode')

    if training_mode == 'scratch':
        if config.get('dataset_type') != 'imagenet':
            raise ValueError(
                "training_mode='scratch' currently only supports dataset_type='imagenet'"
            )

        if config.get('perturb_type') == 'random_target':
            raise ValueError(
                "perturb_type='random_target' requires continuous embedding targets; "
                "and is only compatible with training_mode='finetune'."
            )

        train_dataset = ImagenetDataset(img_dir=config.get('img_dir'), split='train')
        val_dataset = ImagenetDataset(img_dir=config.get('img_dir'), split='val')
        split_info = {'split_type': 'imagenet_predefined'}

        logger.info(
            f"ImageNet predefined splits: "
            f"{len(train_dataset):,} train / {len(val_dataset):,} val images"
        )

        if wandb.run is not None:
            wandb.config.update({
                'dataset_size': len(train_dataset) + len(val_dataset),
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'train_portion': None,
            })

        # Build a THINGS inference dataset for behavioral RSA if paths are provided
        rsa_annotations_file = config.get('rsa_annotations_file')
        rsa_img_dir = config.get('rsa_things_img_dir')
        if config.get('behavioral_rsa') and rsa_annotations_file and rsa_img_dir:
            inference_dataset = ThingsBehavioralDataset(
                img_annotations_file=rsa_annotations_file,
                img_dir=rsa_img_dir,
            )
            logger.info(f"Inference dataset (THINGS): {len(inference_dataset):,} images")
        else:
            inference_dataset = None

        return train_dataset, val_dataset, split_info, (None, None), inference_dataset

    if config.get('dataset_type') == 'things':
        dataset = ThingsBehavioralDataset(
            img_annotations_file=config.get('img_annotations_file'),
            img_dir=config.get('img_dir'),
        )

    elif config.get('dataset_type') == 'imagenet':
        raise ValueError(
            "training_mode='finetune' currently only supports dataset_type='things'"
        )

    else:
        raise ValueError(f"Dataset type {config.get('dataset_type')} not supported")

    if config.get('behavioral_rsa'):
        inference_dataset = ThingsBehavioralDataset(
            img_annotations_file=config.get('rsa_annotations_file'),
            img_dir=config.get('rsa_things_img_dir'),
        )
        
    global_target_mean = None
    global_target_std = None
    
    if config.get('perturb_type') == 'random_target':
        embeddings = dataset.annotations.iloc[:, 1:].values.astype('float32')
        global_target_mean = torch.tensor(np.mean(embeddings), dtype=torch.float32)
        global_target_std = torch.tensor(np.std(embeddings), dtype=torch.float32)
    
    # Determine if we should reuse an existing split
    baseline_checkpoint_path = config.get('baseline_checkpoint_path')
    split_indices_path = None
    if baseline_checkpoint_path:
        split_candidate = os.path.join(
            baseline_checkpoint_path, 'random_states', 'dataset_split_indices.pth'
        )
        if os.path.isfile(split_candidate):
            split_indices_path = split_candidate
    
    # Load or create dataset split
    if split_indices_path:
        if not os.path.isfile(split_indices_path):
            raise FileNotFoundError(f"dataset_split_indices_path not found: {split_indices_path}")
        split_info = torch.load(split_indices_path)
        val_key = 'val_indices' if 'val_indices' in split_info else 'test_indices'
        train_dataset = Subset(dataset, split_info['train_indices'])
        val_dataset  = Subset(dataset, split_info[val_key])
        logger.info(f"Loaded dataset split indices from {split_indices_path}")
    else:
        train_size = int(config['train_portion'] * len(dataset))
        val_size  = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        split_info = {
            'train_indices': list(train_dataset.indices),
            'val_indices':  list(val_dataset.indices),
            'random_seed':   config['random_seed'],
            'train_portion': config['train_portion']
        }

    if wandb.run is not None:
        wandb.config.update({
            'dataset_size':  len(dataset),
            'train_size':    len(train_dataset),
            'val_size':      len(val_dataset),
            'train_portion': config['train_portion']
        })
    
    return train_dataset, val_dataset, split_info, (global_target_mean, global_target_std), inference_dataset


def create_dataloaders(train_dataset, val_dataset, inference_dataset, config):
    """
    Create DataLoader instances for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary with batch_size and random_seed
        
    Returns:
        tuple: (train_loader, val_loader, inference_loader, dataloader_generator)
    """
    cuda_available = torch.cuda.is_available()

    # num_workers: parallel image loading workers per DataLoader.
    # 4 per GPU is a good starting point; override via config if needed.
    num_gpus = torch.cuda.device_count() if cuda_available else 1
    num_workers = config.get('num_workers', 4 * max(num_gpus, 1))

    # pin_memory: pages host tensors into non-pageable memory so the
    # CUDA DMA engine can transfer them without a CPU copy.
    pin_memory = cuda_available

    # persistent_workers: keeps worker processes alive between epochs,
    # avoiding the fork/init overhead at the start of every epoch.
    persistent_workers = num_workers > 0

    # prefetch_factor: how many batches each worker pre-loads ahead of
    # the training loop consuming them.
    prefetch_factor = config.get('prefetch_factor', 2) if num_workers > 0 else None

    # Create generator for reproducible DataLoader shuffling
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(config['random_seed'])

    # IterableDatasets shuffle internally (PyTorch rejects shuffle=True on them)
    # and must use persistent_workers=False so that set_epoch() updates
    # propagate when workers are re-forked at each epoch boundary.
    train_is_iterable = isinstance(train_dataset, IterableDataset)
    val_is_iterable = isinstance(val_dataset, IterableDataset)

    train_loader_kwargs = dict(
        batch_size=config['batch_size'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    if train_is_iterable:
        train_loader_kwargs['shuffle'] = False
        train_loader_kwargs['persistent_workers'] = False
    else:
        train_loader_kwargs['shuffle'] = True
        train_loader_kwargs['generator'] = dataloader_generator
        train_loader_kwargs['persistent_workers'] = persistent_workers

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader_kwargs = dict(
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    val_loader_kwargs['persistent_workers'] = False if val_is_iterable else persistent_workers

    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    if inference_dataset is not None:
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=min(num_workers, 4),
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        inference_loader = None

    return train_loader, val_loader, inference_loader, dataloader_generator


def get_device(cuda_config, logger):
    """Return the torch device for ``cuda_config`` (-1 for all GPUs, else a GPU index)."""
    if torch.cuda.is_available():
        if cuda_config == -1:
            device = torch.device("cuda")
            logger.info(f"Using all {torch.cuda.device_count()} GPUs")
        elif cuda_config == 0:
            device = torch.device("cuda:0")
            logger.info(f"Using GPU 0")
        elif cuda_config == 1:
            device = torch.device("cuda:1")
            logger.info(f"Using GPU 1")
        else:
            device = torch.device("cuda")
            logger.info(f"Using all {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU")
    logger.info(f"Using device: {device}")
    return device


def compute_behavioral_rsa(
    model,
    data_loader,
    device,
    annotations_file,
    distance_metric,
    similarity_metric,
    logger=None,
    epoch_tag=None,
):
    """
    Compute behavioral RSA between model embeddings and target embeddings
    on the provided data_loader.

    For classification models (e.g. ResNet50 trained from scratch), we extract
    penultimate-layer features via ``forward_features`` when available (all
    timm models support this), falling back to the full forward pass otherwise.
    For DataParallel-wrapped models the underlying module is used.
    """
    log = logger.info if logger else print

    # Unwrap DataParallel to access timm's forward_features
    base_model = model.module if hasattr(model, 'module') else model
    use_features = hasattr(base_model, 'forward_features')
    if use_features:
        log("Behavioral RSA: using penultimate-layer features (forward_features)")

    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for _, (_, images, target) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            if use_features:
                feats = base_model.forward_features(images)
                # Collapse any spatial / token dims down to a 1-D vector per image:
                #   [B, C, H, W]  → global avg pool → [B, C]   (CNN)
                #   [B, tokens, D] → mean over tokens → [B, D]  (ViT)
                if feats.dim() == 4:
                    feats = feats.mean(dim=(2, 3))
                elif feats.dim() == 3:
                    feats = feats.mean(dim=1)
                preds.append(feats.cpu())
            else:
                preds.append(model(images).cpu())
            targets.append(target.cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Import RSA utilities lazily to avoid circular imports with inference_core.
    from temporal_perturbations.inference.inference_core import (
        compute_model_rdm,
        compute_rdm_similarity,
    )

    model_rdm = compute_model_rdm(
        preds, dataset_name="things", annotations_file=annotations_file,
        categories=None, distance_metric=distance_metric
    )
    target_rdm = compute_model_rdm(
        targets, dataset_name="targets", annotations_file=None,
        categories=None, distance_metric=distance_metric
    )

    model_ut_idx = np.triu_indices_from(model_rdm, k=1)
    target_ut_idx = np.triu_indices_from(target_rdm, k=1)
    model_ut = model_rdm[model_ut_idx]
    target_ut = target_rdm[target_ut_idx]

    corr, pval = compute_rdm_similarity(model_ut, target_ut, similarity_metric)
    log(f"Behavioral RSA: corr={corr:.4f}, p={pval:.4g}")
    if wandb.run is not None:
        wandb.log({
            'rsa_behavioral_corr': corr,
            'rsa_behavioral_p': pval,
            'epoch': epoch_tag,
        })
    return corr, pval


def save_model_checkpoint(model, checkpoint_dir: str, epoch: int, log_fn=None):
    """
    Save a full model state_dict for from-scratch training.
 
    Handles ``DataParallel`` wrappers transparently by saving the underlying
    ``module`` state_dict so checkpoints are portable across GPU configs.
 
    Args:
        model:          Model (possibly DataParallel-wrapped) to checkpoint.
        checkpoint_dir: Directory to write the file into.
        epoch:          Current epoch index (used in the filename).
        log_fn:         Callable for logging; defaults to ``print``.
    """
    log = log_fn or print
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    checkpoint_file = os.path.join(checkpoint_dir, f"epoch{epoch}_model.pth")
    torch.save(state_dict, checkpoint_file)
    log(f"Model checkpoint saved: {checkpoint_file}")


def _log_resume_debug(
    logger,
    optimizer,
    scheduler,
    state_payload,
    dataloader_generator,
    epoch_loaded: int,
    resume_next_epoch: int,
) -> None:
    """Compact, non-invasive logs for verifying resume (no RNG consumption)."""
    if logger is None:
        return
    logger.info(
        "[debug_resume] loaded checkpoint epoch=%s; training will start at epoch=%s",
        epoch_loaded,
        resume_next_epoch,
    )
    logger.info(
        "[debug_resume] random_states file keys: %s",
        sorted(state_payload.keys()),
    )
    for i, group in enumerate(optimizer.param_groups):
        extra = {k: v for k, v in group.items() if k != "params"}
        logger.info(
            "[debug_resume] optimizer param_groups[%d] lr=%s other=%s",
            i,
            group.get("lr"),
            extra,
        )
    opt_sd = optimizer.state_dict()
    logger.info(
        "[debug_resume] optimizer state: %d param entries in state dict",
        len(opt_sd.get("state", {})),
    )
    if scheduler is not None:
        sd = scheduler.state_dict()
        logger.info(
            "[debug_resume] scheduler %s state_dict keys: %s",
            type(scheduler).__name__,
            list(sd.keys()),
        )
        if "last_epoch" in sd:
            logger.info("[debug_resume] scheduler last_epoch=%s", sd["last_epoch"])
        if "base_lrs" in sd:
            logger.info("[debug_resume] scheduler base_lrs=%s", sd["base_lrs"])
        get_last_lr = getattr(scheduler, "get_last_lr", None)
        if callable(get_last_lr):
            try:
                logger.info("[debug_resume] scheduler.get_last_lr()=%s", get_last_lr())
            except Exception as exc:
                logger.info("[debug_resume] scheduler.get_last_lr() failed: %s", exc)
    dl_restored = (
        dataloader_generator is not None
        and "dataloader_generator_state" in state_payload
    )
    logger.info("[debug_resume] dataloader generator state restored: %s", dl_restored)
    trs = state_payload.get("torch_rng_state")
    if trs is not None:
        logger.info(
            "[debug_resume] torch_rng_state type=%s numel/len=%s",
            type(trs).__name__,
            getattr(trs, "numel", lambda: len(trs))(),
        )


def load_checkpoint_and_states(
    architecture, model, optimizer, dataloader_generator, checkpoint_path,
    random_state_path, logger=None, scheduler=None, debug_resume: bool = False,
    ):
    """
    Load model weights and restore all random states for reproducibility.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to restore state
        dataloader_generator: Generator for DataLoader shuffling
        checkpoint_path: Path to checkpoint directory
        random_state_path: Path to random state file
        logger: Logger instance
        training_mode: Training mode ('scratch' or 'finetune')
        scheduler: Scheduler to restore state
        debug_resume: If True, emit extra resume verification logs (from config ``debug_resume``).
    Returns:
        int: Next epoch to resume training from
    """
    epoch_num = int(random_state_path.split('epoch')[-1].split('_')[0])

    if architecture == 'CLIP-HBA':
        # Load DoRA weights
        loaded_dora_path = load_dora_checkpoint(
            model,
            checkpoint_root=checkpoint_path,
            epoch=epoch_num,
        )
        logger.info(f"Loaded DoRA parameters from: {loaded_dora_path}")

    else:
        model_file = os.path.join(
            checkpoint_path, 'model_checkpoints', f'epoch{epoch_num}_model.pth'
        )
        load_model_checkpoint(model, model_file, logger)
        logger.info(f"Loaded model weights from: {model_file}")
    
    if not os.path.isfile(random_state_path):
        raise FileNotFoundError(f"Missing random state file: {random_state_path}")
    
    state_payload = torch.load(random_state_path, map_location='cpu', weights_only=False)
    
    torch.set_rng_state(state_payload['torch_rng_state'])
    np_set_state(state_payload['numpy_rng_state'])
    random.setstate(state_payload['python_rng_state'])
    
    if torch.cuda.is_available():
        if 'cuda_rng_state' in state_payload:
            torch.cuda.set_rng_state(state_payload['cuda_rng_state'])
        if 'cuda_rng_state_all' in state_payload:
            torch.cuda.set_rng_state_all(state_payload['cuda_rng_state_all'])
    
    if dataloader_generator is not None and 'dataloader_generator_state' in state_payload:
        dataloader_generator.set_state(state_payload['dataloader_generator_state'])
    
    if 'optimizer_state_dict' in state_payload:
        optimizer.load_state_dict(state_payload['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in state_payload:
        scheduler.load_state_dict(state_payload['scheduler_state_dict'])

    logger.info(f"Restored optimizer and RNG states from {random_state_path}")

    next_epoch = epoch_num + 1
    if debug_resume:
        _log_resume_debug(
            logger,
            optimizer,
            scheduler,
            state_payload,
            dataloader_generator,
            epoch_loaded=epoch_num,
            resume_next_epoch=next_epoch,
        )

    return next_epoch


def load_model_checkpoint(model, checkpoint_file: str, logger=None):
    """
    Load a full model state_dict from a checkpoint path.

    Matches ``save_model_checkpoint``: multi-GPU checkpoints store inner-module
    keys (no ``module.`` prefix). ``DataParallel`` loads must use
    ``model.module`` in that case.

    This is used both during training resumption and standalone inference.
    """
    log = logger.info if logger else print
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    wrapped = hasattr(model, "module")
    keys = list(state_dict.keys())
    ckpt_has_module_prefix = bool(keys) and keys[0].startswith("module.")

    if wrapped:
        if ckpt_has_module_prefix:
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
    else:
        if ckpt_has_module_prefix:
            stripped = {
                k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }
            model.load_state_dict(stripped)
        else:
            model.load_state_dict(state_dict)

    log(f"Loaded model weights from: {checkpoint_file}")


def handle_checkpoint_resumption(config, model, optimizer, dataloader_generator, 
logger, scheduler=None):
    """
    Handle checkpoint loading for either explicit resumption or baseline loading.
    
    Args:
        config: Configuration dictionary
        model: Model instance
        optimizer: Optimizer instance
        dataloader_generator: DataLoader generator
        logger: Logger instance
        scheduler: Optional LR scheduler to restore.

    Returns:
        int: Epoch to start/resume from
    """
    resume_epoch = 0
    architecture = config.get("architecture")
    resume_checkpoint_path = config.get("resume_checkpoint_path")
    resume_from_epoch = config.get("resume_from_epoch")
    
    # Case 1: Explicit resumption from a specific checkpoint
    if resume_checkpoint_path and resume_from_epoch is not None:
        random_state_file = os.path.join(
            resume_checkpoint_path,
            'random_states',
            f'epoch{resume_from_epoch}_random_states.pth',
        )
        resume_epoch = load_checkpoint_and_states(
            architecture, model, optimizer, dataloader_generator,
            resume_checkpoint_path, random_state_file, logger,
            scheduler=scheduler,
            debug_resume=bool(config.get("debug_resume", False)),
        )
    
    # Case 2: Load baseline checkpoint before applying perturbation
    elif config.get('perturb_epoch') > 0:
        baseline_epoch = config['perturb_epoch'] - 1
        baseline_checkpoint_path = config.get('baseline_checkpoint_path')
        
        if not baseline_checkpoint_path:
            raise ValueError(
                "baseline_checkpoint_path must be provided when perturb_epoch > 0."
            )
        
        random_state_file = os.path.join(
            baseline_checkpoint_path,
            'random_states',
            f'epoch{baseline_epoch}_random_states.pth',
        )
        resume_epoch = load_checkpoint_and_states(
            architecture, model, optimizer, dataloader_generator, baseline_checkpoint_path,
            random_state_file, logger, scheduler=scheduler,
            debug_resume=bool(config.get("debug_resume", False)),
        )
        logger.info(f"Loaded baseline checkpoint from epoch {baseline_epoch}")
    
    return resume_epoch


def train_one_epoch(model, train_loader, device, optimizer, criterion, 
                   perturb_strategy, epoch_idx, logger, log_interval=10, debug_logging=False):
    """
    Train model for a single epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        device: Device to run training on
        optimizer: Optimizer instance
        criterion: Loss function
        perturb_strategy: Perturbation strategy to apply
        epoch_idx: Current epoch index
        logger: Logger instance
        log_interval: Interval to log metrics with wandb
        debug_logging: Whether to log debug information to the console
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    batch_losses = []
    
    # Log if perturbation is active this epoch
    if perturb_strategy.is_active_epoch(epoch_idx):
        logger.info(f"Applying {perturb_strategy.__class__.__name__} perturbation during epoch {epoch_idx}")
        if wandb.run is not None:
            wandb.log({'perturbation_active': 1, 'epoch': epoch_idx})
    else:
        if wandb.run is not None:
            wandb.log({'perturbation_active': 0, 'epoch': epoch_idx})
    
    progress_bar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader), 
        desc=f"Epoch {epoch_idx}"
    )
    
    for batch_idx, (image_names, images, targets) in progress_bar:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if debug_logging is True and batch_idx == 0:
            print("\n=== DEBUG: BEFORE PERTURBATION ===")
            print("targets (first 10 values in first 10 rows):", targets[:10, :10])
            print("targets shape:", targets.shape)
            print("images (first 10):", images[:10])
            print("images shape:", images.shape)
            print("image_names (first 10):", image_names[:10])
            print("image_names shape:", len(image_names))

        images_before = images.clone()
        targets_before = targets.clone()
        
        # Apply perturbation if active
        images, targets = perturb_strategy.apply_to_batch(
            images, targets, device, epoch_idx, batch_idx
        )

        if debug_logging is True and batch_idx == 0:
            print("\n=== DEBUG: AFTER PERTURBATION ===")
            print("targets (first 10 values in first 10 rows):", targets[:10, :10])
            print("targets shape:", targets.shape)
            print("images (first 10):", images[:10])
            print("images shape:", images.shape)
            print("image_names (first 10):", image_names[:10])
            print("image_names shape:", len(image_names))

            # Numeric check: how much changed?
            delta_targets = (targets - targets_before).abs().mean()
            print("mean |target change|:", delta_targets.item())
            delta_images = (images - images_before).abs().mean()
            print("mean |image change|:", delta_images.item())
            same_rows = (targets == targets_before).all(dim=1).sum().item()
            print("num target vectors unchanged:", same_rows, "/", targets.shape[0])

        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += loss.item() * images.size(0)
        batch_losses.append(batch_loss)
        progress_bar.set_postfix({'loss': batch_loss})

        if batch_idx % log_interval == 0 and wandb.run is not None:
            wandb.log({
                'batch_loss': batch_loss,
                'batch': epoch_idx * len(train_loader) + batch_idx,
                'epoch': epoch_idx,
            })
    
    avg_train_loss = total_loss / len(train_loader.dataset)

    if wandb.run is not None:
        wandb.log({
            'train_loss': avg_train_loss,
            'train_loss_std': np.std(batch_losses),
            'epoch': epoch_idx,
        })

    return avg_train_loss


def save_epoch_results(epoch, train_loss, val_loss, results_writer, rsa_corr=None, rsa_p=None):
    """
    Save training metrics to CSV file.
    
    Args:
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        results_writer: csv.writer for the already-open training results file
        rsa_corr: Optional behavioral RSA correlation
        rsa_p: Optional p-value for behavioral RSA correlation
    """
    data_row = [epoch, train_loss, val_loss]
    if rsa_corr is not None and rsa_p is not None:
        data_row.extend([rsa_corr, rsa_p])
    results_writer.writerow(data_row)


def _replay_early_stop_counter(csv_path, early_stop_start_epoch):
    """Reconstruct (best_val_loss, best_epoch, epochs_no_improve, last_epoch) by
    replaying the gated early-stopping logic over a run's recorded val losses.

    Used when resuming a run that predates early_stop_state.json persistence.
    Seeds the running best with +inf so the first recorded epoch establishes it;
    this can only ever under-count (resume trains slightly longer), never stop
    early. Matches an uninterrupted run exactly whenever the first recorded
    epoch's val loss is <= the model's initial (pre-training) val loss, which
    holds for the perturbation runs here.
    """
    best = float('inf')
    best_epoch = -1
    counter = 0
    last_epoch = -1
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                ep = int(row['epoch'])
                val = float(row['val_loss'])
            except (KeyError, ValueError):
                continue
            if val < best:
                best, best_epoch, counter = val, ep, 0
            elif ep >= early_stop_start_epoch:
                counter += 1
            last_epoch = ep
    return best, best_epoch, counter, last_epoch


def train_model(
    model,
    train_loader,
    val_loader,
    inference_loader,
    device,
    optimizer,
    criterion,
    epochs,
    training_results_save_path,
    save_path,
    random_state_save_path,
    logger=None,
    early_stopping_patience=5,
    dataloader_generator=None,
    vision_layers=1,
    transformer_layers=1,
    perturb_strategy=None,
    start_epoch=0,
    behavioral_rsa=False,
    rsa_annotations_file=None,
    model_rdm_distance_metric="pearson",
    rsa_similarity_metric="spearman",
    debug_logging=False,
    scheduler=None,
    training_mode='finetune',
):
    """
    Main training loop with logging, checkpointing, and early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        inference_loader: Inference data loader
        device: Device for training
        optimizer: Optimizer instance
        criterion: Loss function
        epochs: Maximum number of epochs
        training_results_save_path: Path to save training metrics CSV
        save_path: Directory for checkpoints
        random_state_save_path: Directory for random state snapshots
        logger: Logger instance (uses print if None)
        early_stopping_patience: Epochs without improvement before stopping
        dataloader_generator: Generator for reproducible shuffling
        vision_layers: Number of vision transformer layers to save
        transformer_layers: Number of text transformer layers to save
        perturb_strategy: Perturbation strategy to apply
        start_epoch: Starting epoch (for resumption)
        behavioral_rsa: Whether to compute behavioral RSA
        rsa_annotations_file: Path to annotations file for behavioral RSA
        model_rdm_distance_metric: Distance metric for model RDM
        rsa_similarity_metric: Similarity metric for behavioral RSA
        debug_logging: Whether to log debug information to the console
    """
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    # Early stopping only begins counting AFTER the perturbation window ends, so
    # a long perturbation can't trip the patience counter before the model has a
    # chance to recover. For perturb_epoch=E, perturb_length=L the window covers
    # epochs [E, E+L-1]; the patience clock starts at epoch E+L.
    _pert_end = getattr(perturb_strategy, 'end_epoch', None)
    early_stop_start_epoch = (_pert_end + 1) if isinstance(_pert_end, int) else 0

    # Persisted early-stopping bookkeeping so a resumed run continues the patience
    # counter (and best-loss reference) instead of restarting it.
    early_stop_state_path = os.path.join(save_path, "early_stop_state.json")

    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    # Initial evaluation
    log("*********************************")
    log("Evaluating initial model")
    best_val_loss = evaluate_model(model, val_loader, device, criterion)
    log(f"Initial Validation Loss: {best_val_loss:.4f}")

    # On resume, restore the early-stopping state so epochs_no_improve /
    # best_val_loss continue rather than reset. Prefer the persisted JSON
    # (future runs); fall back to replaying the recorded CSV (runs that predate
    # state persistence, e.g. the early-truncated rows).
    if start_epoch > 0:
        restored = False
        if os.path.exists(early_stop_state_path):
            try:
                with open(early_stop_state_path) as _f:
                    _es = json.load(_f)
                if int(_es.get("last_completed_epoch", -1)) == start_epoch - 1:
                    best_val_loss = float(_es["best_val_loss"])
                    epochs_no_improve = int(_es["epochs_no_improve"])
                    best_epoch = int(_es.get("best_epoch", -1))
                    restored = True
                    log(
                        f"Restored early-stopping state (json): best_val_loss={best_val_loss:.4f} "
                        f"(epoch {best_epoch}), epochs_no_improve={epochs_no_improve}"
                    )
            except Exception as _es_err:
                log(f"Could not read early_stop_state.json ({_es_err}); will reconstruct from CSV.")
        if not restored and os.path.exists(training_results_save_path):
            _b, _be, _c, _le = _replay_early_stop_counter(
                training_results_save_path, early_stop_start_epoch
            )
            if _le == start_epoch - 1:
                best_val_loss, best_epoch, epochs_no_improve = _b, _be, _c
                restored = True
                log(
                    f"Reconstructed early-stopping state (CSV replay): best_val_loss={best_val_loss:.4f} "
                    f"(epoch {best_epoch}), epochs_no_improve={epochs_no_improve}"
                )
            else:
                log(
                    f"CSV last epoch {_le} != resume epoch {start_epoch - 1}; "
                    "starting patience counter fresh."
                )
        if not restored:
            log("No early-stopping state to restore; starting patience counter fresh.")
    # Optional behavioral RSA at initial evaluation
    if behavioral_rsa:
        try:
            compute_behavioral_rsa(
                model=model,
                data_loader=inference_loader,
                device=device,
                annotations_file=rsa_annotations_file,
                distance_metric=model_rdm_distance_metric,
                similarity_metric=rsa_similarity_metric,
                logger=logger,
                epoch_tag=-1,
            )
        except Exception as rsa_err:
            if logger:
                logger.warning(f"Initial behavioral RSA failed: {rsa_err}")
            else:
                print(f"Initial behavioral RSA failed: {rsa_err}")
    log("*********************************\n")

    if wandb.run is not None:
        wandb.log({
            'initial_val_loss': best_val_loss,
            'epoch': -1,
        })
    
    # Create directories for outputs
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(training_results_save_path), exist_ok=True)
    
    # Open the CSV file once for the entire training run to avoid repeated
    # open/close calls on network filesystems (e.g. CIFS/SMB), which can
    # fail with EINVAL after many round-trips.
    csv_mode = 'w' if not (start_epoch > 0 and os.path.exists(training_results_save_path)) else 'a'
    results_file = open(training_results_save_path, csv_mode, newline='')
    results_writer = csv.writer(results_file)
    if csv_mode == 'w':
        header = ['epoch', 'train_loss', 'val_loss']
        if behavioral_rsa:
            header.extend(['rsa_behavioral_corr', 'rsa_behavioral_p'])
        results_writer.writerow(header)

    # Main training loop
    for epoch in range(start_epoch, epochs):
        # Reseed streaming dataset shuffle for this epoch (IterableDataset path)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        # Train one epoch
        avg_train_loss = train_one_epoch(
            model, train_loader, device, optimizer, criterion,
            perturb_strategy, epoch, logger, debug_logging=debug_logging,
        )
        
        # Evaluate on validation set
        avg_val_loss = evaluate_model(model, val_loader, device, criterion)
        
        rsa_corr = None
        rsa_p = None

        # Log results
        print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        log(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        if perturb_strategy.is_active_epoch(epoch):
            logger.info(f"*** Perturbation '{perturb_strategy.__class__.__name__}' was applied during epoch {epoch} ***")

        if scheduler is not None:
            scheduler.step()

        if wandb.run is not None:
            wandb.log({
                'val_loss': avg_val_loss,
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        # Optional behavioral RSA at end of epoch
        if behavioral_rsa:
            try:
                rsa_corr, rsa_p = compute_behavioral_rsa(
                    model=model,
                    data_loader=inference_loader,
                    device=device,
                    annotations_file=rsa_annotations_file,
                    distance_metric=model_rdm_distance_metric,
                    similarity_metric=rsa_similarity_metric,
                    logger=logger,
                    epoch_tag=epoch,
                )
            except Exception as rsa_err:
                if logger:
                    logger.warning(f"Behavioral RSA failed at epoch {epoch}: {rsa_err}")
                else:
                    print(f"Behavioral RSA failed at epoch {epoch}: {rsa_err}")
        
        # Save metrics to CSV; flush to ensure each epoch is durably written
        save_epoch_results(epoch, avg_train_loss, avg_val_loss, results_writer, rsa_corr, rsa_p)
        results_file.flush()

        # Save random states for reproducibility
        save_random_states(
            optimizer, epoch, random_state_save_path, 
            dataloader_generator, logger=logger, scheduler=scheduler,
        )
        
        # Save model checkpoint
        if training_mode == 'scratch':
            checkpoint_dir = os.path.join(save_path, "model_checkpoints")
            save_model_checkpoint(model, checkpoint_dir, epoch, log_fn=log)
        else:
            checkpoint_dir = os.path.join(save_path, "dora_params")
            save_dora_parameters(
                model, checkpoint_dir, epoch, 
                vision_layers, transformer_layers, log_fn=log
            )
        log(f"Checkpoint saved for epoch {epoch}")

        if wandb.run is not None:
            save_to_wandb = (epoch < 20) or (epoch % 5 == 0)
            if save_to_wandb:
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-epoch-{epoch}",
                    type="model",
                    description=f"Model checkpoint at epoch {epoch}",
                    metadata={
                        'epoch': epoch,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                    }
                )
                artifact.add_dir(checkpoint_dir)
                artifact.add_file(training_results_save_path)
                wandb.log_artifact(artifact)

        # Early stopping check — the patience counter stays frozen until the
        # perturbation window has fully ended (see early_stop_start_epoch).
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if wandb.run is not None:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch
        elif epoch >= early_stop_start_epoch:
            epochs_no_improve += 1

        # Persist early-stopping bookkeeping so a resume from this epoch continues
        # the counter instead of resetting it.
        with open(early_stop_state_path, "w") as _f:
            json.dump(
                {
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "epochs_no_improve": epochs_no_improve,
                    "last_completed_epoch": epoch,
                },
                _f,
            )

        if wandb.run is not None:
            wandb.log({
                'epochs_no_improve': epochs_no_improve,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
            })

        if epoch >= early_stop_start_epoch and epochs_no_improve >= early_stopping_patience:
            log("\n\n*********************************")
            log(f"Early stopping triggered at epoch {epoch}")
            log("*********************************\n\n")
            if wandb.run is not None:
                wandb.run.summary["stopped_early"] = True
                wandb.run.summary["final_epoch"] = epoch
            break

    results_file.close()

    if epochs_no_improve < early_stopping_patience:
        if wandb.run is not None:
            wandb.run.summary["stopped_early"] = False
            wandb.run.summary["final_epoch"] = epochs

    # Log full training metrics CSV once at completion
    if wandb.run is not None:
        try:
            metrics_artifact = wandb.Artifact(
                name="training-results-final",
                type="metrics",
                description="Full training results CSV",
                metadata={
                    'best_val_loss': best_val_loss,
                    'final_epoch': wandb.run.summary.get("final_epoch", epochs),
                }
            )
            metrics_artifact.add_file(training_results_save_path)
            wandb.log_artifact(metrics_artifact)
        except Exception as art_err:
            if logger:
                logger.warning(f"Failed to log training results artifact: {art_err}")
            else:
                print(f"Failed to log training results artifact: {art_err}")


def run_training_experiment(config):
    """
    Main entry point for running a training experiment.
    
    Supports two training modes controlled by ``config['training_mode']``:
 
    * ``'scratch'``  – Train a timm ViT or ResNet from scratch on ImageNet
      (CrossEntropyLoss, full model checkpoints, no DoRA).
    * ``'finetune'`` – Fine-tune a CLIP-HBA model with DoRA adaptation
      (MSELoss, DoRA checkpoints, optional behavioral RSA).
 
    Both modes support all perturbation types from
    ``perturbation_utils.py`` (except ``random_target`` which requires
    continuous embedding targets and is therefore only valid for
    ``training_mode='finetune'``).
    
    Args:
        config: Configuration dictionary containing all training parameters.
    """
    training_mode = config.get("training_mode")

    seed_everything(config['random_seed'])
    
    # Build run name once so wandb and filesystem stay in sync
    run_name = get_run_name(config)
    config['wandb_run_name'] = run_name
    
    # Align save_path with the run name (append unless already present)
    base_save_path = config.get('save_path', '')
    if base_save_path:
        normalized_base = os.path.normpath(base_save_path)
        if os.path.basename(normalized_base) != run_name:
            save_path = os.path.join(base_save_path, run_name)
        else:
            save_path = base_save_path
    else:
        save_path = run_name
    config['save_path'] = save_path
    
    os.makedirs(config['save_path'], exist_ok=True)
    perturb_type = config.get('perturb_type')
    if perturb_type:
        log_file = os.path.join(config['save_path'], f'training_log_{perturb_type}.txt')
    else:
        log_file = os.path.join(config['save_path'], 'training_log.txt')
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("Starting Training Run")
    logger.info(f"Training mode : {training_mode}")
    logger.info(f"Architecture  : {config.get('architecture', 'unknown')}")
    logger.info(f"Log file      : {log_file}")
    logger.info("=" * 80)
    
    # Persist config snapshots inside the run folder
    resolved_cfg_path = os.path.join(save_path, "resolved_config.yaml")
    input_cfg_path    = os.path.join(save_path, "training_config_snapshot.yaml")
    with open(resolved_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    # Input snapshot is the same available config dict; retained for provenance
    with open(input_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    
    save_path, training_results_save_path, random_state_save_path = setup_paths(config)

    # Initialize wandb before any wandb.config updates in dataset setup
    if not config.get('no_wandb'):
        init_wandb(config, resume_epoch=0)
    
    train_dataset, val_dataset, split_info, target_stats, inference_dataset = setup_dataset(config, logger)
    global_target_mean, global_target_std = target_stats
    
    split_file = os.path.join(random_state_save_path, 'dataset_split_indices.pth')
    os.makedirs(random_state_save_path, exist_ok=True)
    torch.save(split_info, split_file)
    logger.info(f"Dataset split info saved: {split_file}")
    
    train_loader, val_loader, inference_loader, dataloader_generator = create_dataloaders(
        train_dataset, val_dataset, inference_dataset, config
    )
    
    device = get_device(config['cuda'], logger)
    
    model = build_model(
        architecture=config.get('architecture'),
        pretrained=config.get('pretrained'),
        clip_hba_backbone=config.get('clip_hba_backbone'),
        vision_layers=config.get('vision_layers'),
        transformer_layers=config.get('transformer_layers'),
        rank=config.get('rank'),
        cuda=config.get('cuda'),
        device=device,
        wandb_watch_model=config.get('wandb_watch_model'),
        wandb_log_freq=config.get('wandb_log_freq'),
        num_classes=config.get('num_classes', 1000),
    )
    model.to(device)
    
    # Resolve total epoch count (max_duration overrides epochs if present)
    epochs = config.get('max_duration')
    config['epochs'] = epochs  # normalise so downstream code sees an int

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, epochs)
    if scheduler is not None:
        logger.info(f"LR scheduler: {scheduler.__class__.__name__}")
    
    # Handle checkpoint resumption if needed
    resume_epoch = handle_checkpoint_resumption(
        config, model, optimizer, dataloader_generator, logger, scheduler=scheduler,
    )

    log_model_architecture(model, logger)
    
    criterion_name = config.get('criterion')
    if criterion_name == 'MSELoss':
        criterion = nn.MSELoss()
    elif criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {criterion_name} not supported")
    
    perturb_strategy = choose_perturbation_strategy(
        config['perturb_type'],
        config['perturb_epoch'],
        config['perturb_length'],
        config['perturb_seed'],
        target_mean=global_target_mean,
        target_std=global_target_std,
    )
    
    logger.info("\nModel Configuration:")
    logger.info("-------------------")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\nTraining layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    
    logger.info(f"\nNumber of trainable parameters: {count_trainable_parameters(model)}\n")
    
    # Run training
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            inference_loader=inference_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            training_results_save_path=training_results_save_path,
            logger=logger,
            early_stopping_patience=config['early_stopping_patience'],
            save_path=save_path,
            random_state_save_path=random_state_save_path,
            dataloader_generator=dataloader_generator,
            vision_layers=config.get('vision_layers'),
            transformer_layers=config.get('transformer_layers'),
            perturb_strategy=perturb_strategy,
            start_epoch=resume_epoch,
            behavioral_rsa=config.get('behavioral_rsa', False),
            rsa_annotations_file=config.get('rsa_annotations_file'),
            model_rdm_distance_metric=config.get('model_rdm_distance_metric', 'pearson'),
            rsa_similarity_metric=config.get('rsa_similarity_metric', 'spearman'),
            debug_logging=config.get('debug_logging', False),
            scheduler=scheduler,
            training_mode=training_mode,
        )
    finally:
        if wandb.run is not None:
            wandb.finish()
        if hasattr(logger, 'restore_streams'):
            logger.restore_streams()
