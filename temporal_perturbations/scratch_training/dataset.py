import os
import random
import logging
import warnings
from io import BytesIO
from typing import Optional, Callable, Dict, Tuple

import numpy as np

# Set TOKENIZERS_PARALLELISM to avoid deadlocks with forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image, ImageFilter, ImageFile
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# --- LitData Imports ---
try:
    import litdata as ld
    import polars  # noqa: F401  (verifies the import path actually works at all)
    # polars-u64-idx provides the `polars` import but registers under a different
    # PyPI distribution name, so litdata's RequirementCache("polars>1.0.0") check
    # fails. Override the private flag in the modules that consult it.
    # If litdata removed/renamed the flag (upstream check was likely fixed in a newer
    # release), the AttributeError below means the workaround is no longer needed —
    # log and continue rather than crash at import time before logging is configured.
    try:
        import litdata.constants as _ld_constants
        import litdata.streaming.item_loader as _ld_item_loader
        _ld_constants._POLARS_AVAILABLE = True
        _ld_item_loader._POLARS_AVAILABLE = True
    except AttributeError as e:
        logger.warning(
            "litdata _POLARS_AVAILABLE override no-oped (%s). "
            "Either litdata fixed the polars detection upstream (good) or the flag "
            "moved (in which case parquet loading may fail later).", e,
        )
    from litdata.streaming.item_loader import ParquetLoader
    LITDATA_AVAILABLE = True
except ImportError:
    LITDATA_AVAILABLE = False
    ParquetLoader = None  # Define dummy for type hints

warnings.filterwarnings("ignore", "(Possibly )?DecompressionBombWarning")

# Samples held out from the YFCC15M stream for validation (see the train/val split below).
VAL_SPLIT_SIZE = 10000


def yfcc15m_train_size(config) -> int:
    """Number of YFCC15M *training* samples (full stream minus the val holdout).

    Reads only the LitData index (cheap, no streaming), mirroring the exact
    input_dir / item_loader / index_path used to build the training dataset, so it
    can size the cosine LR horizon before the datamodule's setup() runs. Returns 0
    if the index can't be read (caller falls back to the num_training_batches path).
    """
    if not LITDATA_AVAILABLE or ParquetLoader is None:
        return 0
    try:
        _input_dir = config.parquet_data_dir if config.parquet_data_dir else config.hf_dataset_name
        _index_kwargs = {}
        if config.dataset_index_path:
            _index_kwargs["index_path"] = config.dataset_index_path
        elif config.parquet_data_dir:
            _index_kwargs["index_path"] = "index.json"
        ds = ld.StreamingDataset(input_dir=_input_dir, item_loader=ParquetLoader(), **_index_kwargs)
        # StreamingDataset.__len__ is PER-RANK under a distributed launch (torchrun sets
        # WORLD_SIZE), so scale back to the global sample count before removing the val
        # holdout — otherwise the horizon is undercounted by world_size (e.g. 8x).
        world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        return max(1, len(ds) * world - VAL_SPLIT_SIZE)
    except Exception:
        return 0


def seed_worker(worker_id):
    """
    Worker seed function for DataLoader to ensure reproducibility in distributed training
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- Gaussian Blur Implementation (from DeCLIP/MoCo) ---
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
# --- End Gaussian Blur ---

def get_transform(image_size: int = 224, is_train: bool = True, interpolation: str = "bicubic"):
    """
    Get the image transform pipeline.
    Uses MOCOv2 augmentations for training, standard CLIP eval transforms otherwise.
    
    Args:
        image_size: Size to resize image to
        is_train: Whether to include data augmentation for training
        interpolation: Interpolation method for resizing (used for eval only)
    """
    interpolation_map = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "nearest": transforms.InterpolationMode.NEAREST
    }
    interpolation_mode = interpolation_map.get(interpolation, transforms.InterpolationMode.BICUBIC)
    
    # Normalize with OpenAI CLIP mean and std (as used in the rest of the project)
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    
    if is_train:
        # MOCO V2 style augmentations (closer to DeCLIP)
        # Reference: https://github.com/facebookresearch/moco/blob/main/main_moco.py
        # DeCLIP config: prototype/data/imagenet_dataloader.py build_common_augmentation('MOCOV2')
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)), # DeCLIP uses 0.2-1.0
            transforms.RandomApply([ # Applied with p=0.8
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) # DeCLIP uses 0.4, 0.4, 0.4, 0.1
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2), # DeCLIP uses p=0.2
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5), # DeCLIP uses p=0.5
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Standard CLIP evaluation transform
        transform = transforms.Compose([
            transforms.Resize(
                image_size, # Use the integer size directly
                interpolation=interpolation_mode
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    return transform

# --- LitData Preprocessing Function ---
def preprocess_yfcc_litdata(
    item: Dict,
    transform: Callable,
    tokenizer: AutoTokenizer, # Add tokenizer argument
    # Removed eda_augmenter argument
    image_col: str = "images",
    text_col: str = "texts",
    context_length: int = 77 # Standard CLIP context length
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Processes a single item yielded by LitData's ParquetLoader for YFCC15M.
    Removed EDA text augmentation.

    Args:
        item: Dictionary containing raw data (image bytes, token list).
        transform: Image transform function.
        tokenizer: Tokenizer instance for decoding/encoding text.
        image_col: Key for image data in the item dict.
        text_col: Key for text token data in the item dict.
        context_length: Target sequence length for token padding/truncation.

    Returns:
        A tuple (image_tensor, text_tensor) or None if processing fails.
    """
    try:
        # ---- Image Processing (Existing Logic) ----
        img_data = item.get(image_col)
        if img_data is None:
            return None
        
        # Handle different image data formats
        if isinstance(img_data, bytes):
            img = Image.open(BytesIO(img_data))
        elif isinstance(img_data, dict) and "bytes" in img_data:
             img = Image.open(BytesIO(img_data["bytes"]))
        elif isinstance(img_data, Image.Image):
             img = img_data
        else:
            return None
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Ensure we have a valid PIL Image before applying transform
        if not isinstance(img, Image.Image):
            return None
            
        image_tensor = transform(img)

        tokens_raw = item.get(text_col)
        if tokens_raw is None or (isinstance(tokens_raw, list) and len(tokens_raw) == 0):
            return None

        # Convert raw tokens (potentially list of floats/ints) to clean list of ints
        try:
             token_ids = [int(t) for t in tokens_raw if isinstance(t, (int, float)) and not np.isnan(t)]
             if not token_ids: return None # Skip if no valid tokens
        except: # Catch potential conversion errors
             return None # Skip if conversion fails

        # Process token_ids directly without decode/re-tokenize step
        try:
            # Handle padding and truncation directly on token_ids
            if len(token_ids) > context_length:
                # Truncate if too long
                token_ids = token_ids[:context_length]
            elif len(token_ids) < context_length:
                # Pad if too short - use tokenizer's pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                token_ids = token_ids + [pad_token_id] * (context_length - len(token_ids))
            
            # Convert to PyTorch tensor
            tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
            
            # Verify final tensor shape
            if tokens_tensor.shape[0] != context_length:
                return None # Skip if shape is wrong

        except Exception as process_e:
            return None # Skip item

        return image_tensor, tokens_tensor

    except Exception as e:
        return None


# --- Custom LitData Dataset Class ---
class YFCC15MLitDataDataset(ld.StreamingDataset):
    """
    Custom StreamingDataset for YFCC15M using LitData.
    Applies preprocessing including image transforms and text tokenization.
    """
    def __init__(
        self,
        *args,
        transform: Callable, # Image transform
        tokenizer: AutoTokenizer, # Tokenizer instance
        # Removed eda_augmenter
        image_col: str = "images",
        text_col: str = "texts",
        context_length: int = 77,
        **kwargs
    ):
        # IMPORTANT: Don't pass transform to base class to avoid litdata 0.2.51 issue
        # where base class applies transform to raw dict instead of PIL image
        super().__init__(*args, **kwargs)
        
        # Store our custom transform separately to avoid base class interference
        self._custom_transform = transform
        self.tokenizer = tokenizer
        # Removed self.eda_augmenter
        self.image_col = image_col
        self.text_col = text_col
        self.context_length = context_length

        self._bad_sample_count = 0  # per-worker counter for skipped samples
        self._last_good_sample = None  # per-worker fallback for bad samples
        # Track which chunk indices have already emitted a bad-sample warning so
        # that a single corrupted chunk (10 k samples) does not flood the log.
        self._warned_chunks: set = set()
        
        # Disable base class transform to prevent it from being applied to raw dict
        # Use identity function instead of None to avoid TypeError
        self.transform = lambda x: x

        if self._custom_transform is None:
             raise ValueError("An image transform function must be provided")
        if self.tokenizer is None:
             raise ValueError("A tokenizer instance must be provided")
        # Removed check for TEXTAUGMENT_AVAILABLE
    
    def set_transform(self, transform: Callable):
        """Method to update transform after dataset creation (e.g., after splitting)"""
        self._custom_transform = transform
    
    def set_preprocessing_params(self, transform: Callable, tokenizer: AutoTokenizer, 
                                image_col: str = None, text_col: str = None, 
                                context_length: int = None):
        """Method to update all preprocessing parameters after dataset creation"""
        self._custom_transform = transform
        self.tokenizer = tokenizer
        if image_col is not None:
            self.image_col = image_col
        if text_col is not None:
            self.text_col = text_col
        if context_length is not None:
            self.context_length = context_length

    def __getitem__(self, index) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # Returning None here would shrink this rank's batch and deadlock the
        # all_gather inside ClipLoss. Fall back to the last good sample seen by
        # this worker so the batch stays full-shape on every rank. The collate
        # also pads as belt-and-suspenders for the cold-start case.
        # BaseException catches pyo3_runtime.PanicException from polars row
        # groups exceeding 32-bit offset limits (install polars-u64-idx to
        # remove the underlying cause).
        try:
            raw_item = super().__getitem__(index)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            self._bad_sample_count += 1
            chunk_id = getattr(index, "chunk_index", None)
            if chunk_id not in self._warned_chunks:
                self._warned_chunks.add(chunk_id)
                logger.warning(
                    f"Read error in this worker – first bad sample from chunk_index={chunk_id} "
                    f"(total bad so far: {self._bad_sample_count}, index={index}): "
                    f"{type(e).__name__}: {e}"
                )
            return self._last_good_sample

        processed_item = preprocess_yfcc_litdata(
            item=raw_item,
            transform=self._custom_transform,
            tokenizer=self.tokenizer,
            image_col=self.image_col,
            text_col=self.text_col,
            context_length=self.context_length,
        )

        if processed_item is None:
            self._bad_sample_count += 1
            chunk_id = getattr(index, "chunk_index", None)
            if chunk_id not in self._warned_chunks:
                self._warned_chunks.add(chunk_id)
                logger.warning(
                    f"Bad sample in this worker – first bad sample from chunk_index={chunk_id} "
                    f"(total bad so far: {self._bad_sample_count}, index={index})"
                )
            return self._last_good_sample

        self._last_good_sample = processed_item
        return processed_item


# --- First add helper functions for cifar100 validation dataset ---
def create_cifar100_validation_dataloader(
    transform, 
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool, 
    distributed: bool = False, 
    val_dataset_root: str = None, 
    seed: int = 42
):
    """Create a CIFAR100 validation dataloader"""
    logger.info("Creating CIFAR100 validation dataloader")
    
    # Default cache path or use config
    cifar100_cache_dir = val_dataset_root or os.path.join(os.getcwd(), 'cifar100_eval_cache')
    os.makedirs(cifar100_cache_dir, exist_ok=True)
    logger.info(f"CIFAR100 cache/download directory: {cifar100_cache_dir}")
    
    try:
        val_dataset = tv_datasets.CIFAR100(
            root=cifar100_cache_dir,
            train=False,  
            download=True,
            transform=transform
        )
        logger.info(f"CIFAR100 test set loaded successfully ({len(val_dataset)} samples).")
        
        val_sampler = None
        if distributed:
            try:
                # Check if process group is initialized before creating DistributedSampler
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed, drop_last=False)
                    logger.info("Using DistributedSampler for CIFAR100 validation")
                else:
                    logger.warning("torch.distributed not initialized when creating val_sampler. Using sequential sampler.")
            except Exception as e:
                logger.warning(f"Error creating DistributedSampler for validation: {e}. Using sequential sampler.")
        
        dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,  # No shuffle for validation
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        logger.info(f"Created CIFAR100 validation DataLoader with batch size {batch_size}.")
        return dataloader
    except Exception as e:
        logger.error(f"Failed to create CIFAR100 validation dataset/dataloader: {e}", exc_info=True)
        logger.warning("Proceeding without CIFAR100 validation.")
        return None

def create_cifar10_validation_dataloader(
    transform, 
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool, 
    distributed: bool = False, 
    val_dataset_root: str = None, 
    seed: int = 42
):
    """Create a CIFAR10 validation dataloader"""
    logger.info("Creating CIFAR10 validation dataloader")
    
    # Default cache path or use config
    cifar10_cache_dir = val_dataset_root or os.path.join(os.getcwd(), 'cifar10_eval_cache')
    os.makedirs(cifar10_cache_dir, exist_ok=True)
    logger.info(f"CIFAR10 cache/download directory: {cifar10_cache_dir}")
    
    try:
        val_dataset = tv_datasets.CIFAR10(
            root=cifar10_cache_dir,
            train=False,  # Use the test set for validation
            download=True,
            transform=transform
        )
        logger.info(f"CIFAR10 test set loaded successfully ({len(val_dataset)} samples).")
        
        val_sampler = None
        if distributed:
            try:
                # Check if process group is initialized before creating DistributedSampler
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed, drop_last=False)
                    logger.info("Using DistributedSampler for CIFAR10 validation")
                else:
                    logger.warning("torch.distributed not initialized when creating val_sampler. Using sequential sampler.")
            except Exception as e:
                logger.warning(f"Error creating DistributedSampler for validation: {e}. Using sequential sampler.")
        
        dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,  # No shuffle for validation
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        logger.info(f"Created CIFAR10 validation DataLoader with batch size {batch_size}.")
        return dataloader
    except Exception as e:
        logger.error(f"Failed to create CIFAR10 validation dataset/dataloader: {e}", exc_info=True)
        logger.warning("Proceeding without CIFAR10 validation.")
        return None

# --- Now modify the create_dataloaders function ---
def create_dataloaders(
    config: 'DataConfig', # Use the config object
    seed: int = 42,
    distributed: bool = True,
    multi_val: bool = False, # New parameter to enable multiple validation datasets
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets based on config.

    Args:
        config: DataConfig object containing dataset and dataloader settings.
        seed: Random seed for reproducibility.
        distributed: Whether to use DistributedSampler for training.
        multi_val: Whether to return multiple validation dataloaders

    Returns:
        Dictionary of dataloaders for each split.
    """
    dataloaders = {}
    g = torch.Generator()
    g.manual_seed(seed)
    
    # --- Initialize Tokenizer ---
    # Moved outside the dataset loop as it's needed for YFCC text processing
    tokenizer = None
    if config.dataset_type == "yfcc15m_litdata":
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            logger.info(f"Tokenizer '{config.tokenizer_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{config.tokenizer_name}': {e}")
            raise RuntimeError(f"Tokenizer '{config.tokenizer_name}' is required for YFCC15M.") from e
        
    use_persistent_workers = config.num_workers > 0 and getattr(config, 'use_persistent_workers', False)
    
    # Common DataLoader kwargs (ensure pin_memory uses the config value)
    dataloader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory, # Use config value here
        "worker_init_fn": seed_worker,
        "persistent_workers": True,
        "prefetch_factor": 2 if config.num_workers > 0 else None # Reduced from 4 to halve per-worker prefetch RAM footprint
    }

    if config.dataset_type == "yfcc15m_litdata":
        if not LITDATA_AVAILABLE:
            raise ImportError("LitData is required for 'yfcc15m_litdata' dataset type. Please install litdata.")
        if tokenizer is None: # Should not happen due to check above, but for safety
             raise RuntimeError("Tokenizer could not be initialized for YFCC15M LitData.")


        logger.info(f"Creating LitData dataloaders for YFCC15M dataset.")
        logger.info(f"  Parquet data directory: {config.parquet_data_dir}")
        # Removed logging about text augmentation

        # --- Define Transforms ---
        train_transform = get_transform(config.image_size, True, config.interpolation)
        eval_transform = get_transform(config.image_size, False, config.interpolation)
        context_length = 77 # Standard CLIP

        # --- Create Custom LitData StreamingDataset for TRAINING ---
        try:
            # We first instantiate the full dataset intended for training
            _input_dir = config.parquet_data_dir if config.parquet_data_dir else config.hf_dataset_name
            # Only supply index_path for local directories; for hf:// URIs litdata
            # resolves the index automatically from input_dir and passing a bare
            # filename causes it to look for the file locally instead.
            _index_kwargs = {}
            if config.dataset_index_path:
                _index_kwargs["index_path"] = config.dataset_index_path
            elif config.parquet_data_dir:
                _index_kwargs["index_path"] = "index.json"
            _max_cache_size = getattr(config, 'litdata_max_cache_size', None)
            full_train_dataset = YFCC15MLitDataDataset(
                input_dir=_input_dir,
                cache_dir=config.hf_cache_dir,
                **({"max_cache_size": _max_cache_size} if _max_cache_size is not None else {}),
                **_index_kwargs,
                item_loader=ParquetLoader(),
                shuffle=True,
                # Don't pass transform to base class - it will be stored as _custom_transform
                transform=train_transform,
                tokenizer=tokenizer,
                image_col=config.image_col,
                text_col=config.text_col,
                context_length=context_length
            )
            logger.info("Initial YFCC15MLitDataDataset created for splitting.")

            # --- Split the dataset into train and validation ---
            val_split_size = VAL_SPLIT_SIZE
            len_full_train_dataset = len(full_train_dataset)
            if len_full_train_dataset > val_split_size:
                 logger.info(f"Splitting {val_split_size} samples for validation from YFCC15M stream len {len_full_train_dataset}...")
                 train_dataset, val_dataset_yfcc = ld.train_test_split(
                     full_train_dataset,
                     splits=[1 - val_split_size / len_full_train_dataset, val_split_size / len_full_train_dataset],
                     seed=seed
                 )
                 logger.info(f"YFCC15M split complete: Train size ~{len(train_dataset)}, Val size {len(val_dataset_yfcc)}")
                 
                 # Fix for litdata 0.2.51: Properly set preprocessing parameters for validation dataset
                 if hasattr(val_dataset_yfcc, 'set_preprocessing_params'):
                     val_dataset_yfcc.set_preprocessing_params(
                         transform=eval_transform,
                         tokenizer=tokenizer,
                         image_col=config.image_col,
                         text_col=config.text_col,
                         context_length=context_length
                     )
                     logger.info("Applied evaluation transform and preprocessing params to YFCC15M validation split using set_preprocessing_params.")
                 elif hasattr(val_dataset_yfcc, 'set_transform'):
                     val_dataset_yfcc.set_transform(eval_transform)
                     logger.info("Applied evaluation transform to YFCC15M validation split using set_transform.")
                 elif hasattr(val_dataset_yfcc, 'item_loader') and hasattr(val_dataset_yfcc.item_loader, 'transform'):
                      val_dataset_yfcc.item_loader.transform = eval_transform
                      logger.info("Applied evaluation transform to YFCC15M validation split via item_loader.")
                 else:
                      logger.warning("Could not modify transform for YFCC15M validation split. It might use training augmentations.")

            else:
                 logger.warning(f"Full dataset size ({len(full_train_dataset)}) is not larger than requested validation size ({val_split_size}). Using full dataset for training and skipping YFCC validation split.")
                 train_dataset = full_train_dataset
                 val_dataset_yfcc = None

        except FileNotFoundError as e:
             # Catch specific error if index is still missing
             logger.error(f"LitData index file ('index.json') likely missing inside the input_dir: {config.parquet_data_dir}")
             logger.error(f"Original error: {e}")
             logger.error("Please run the indexing script (e.g., index_dataset.py) first.")
             raise
        except Exception as e:
            logger.error(f"Failed to create YFCC15MLitDataDataset: {e}", exc_info=True)
            raise

        # --- Create LitData StreamingDataLoader for TRAINING ---
        # Use collate function to handle potential IndexErrors from __getitem__
        def skip_indexing_error_collate(batch):
             # Pad to len(batch) by duplicating valid items so every rank's tensor
             # has identical shape — required by the all_gather in ClipLoss.
             expected_size = len(batch)
             filtered_batch = [item for item in batch if item is not None]
             if not filtered_batch:
                 return None
             if len(filtered_batch) < expected_size:
                 pad = [filtered_batch[i % len(filtered_batch)]
                        for i in range(expected_size - len(filtered_batch))]
                 filtered_batch = filtered_batch + pad
             try:
                 return torch.utils.data.dataloader.default_collate(filtered_batch)
             except Exception:
                 return None

        if train_dataset: # Only create loader if train_dataset exists
            dataloaders["train"] = ld.StreamingDataLoader(
                train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=use_persistent_workers,
                prefetch_factor=dataloader_kwargs.get("prefetch_factor", 8),
                collate_fn=skip_indexing_error_collate # Use the custom collate
            )
            logger.info(f"LitData StreamingDataLoader created for training.")
        else:
             dataloaders["train"] = None
             logger.error("Failed to create training dataset/dataloader for YFCC15M.")


        # --- Create validation dataloaders ---
        eval_batch_size = config.eval_batch_size if config.eval_batch_size else config.batch_size // 2
        val_dataloaders = [] # Initialize empty list

        # --- Create YFCC15M Validation DataLoader (if split was successful) ---
        # Materialize the val split into a MapDataset and use a torch
        # DistributedSampler. The previous `ld.StreamingDataLoader(val_dataset_yfcc, ...)`
        # path didn't partition val samples evenly across DDP ranks: some ranks
        # logged `val/yfcc15m/clip_loss` and others didn't, which produced
        # divergent metric-key sets across ranks and a hang at the validation
        # epoch_end barrier (rank 2 BARRIER vs rank 0 GATHER, SeqNum mismatch).
        if val_dataset_yfcc:
            world_size = torch.distributed.get_world_size() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 1
            rank = torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0

            logger.info(f"[rank {rank}] Materializing YFCC15M val split into memory ...")
            val_items = []
            for item in val_dataset_yfcc:
                if item is None:
                    continue
                val_items.append(item)
            logger.info(f"[rank {rank}] Materialized {len(val_items)} YFCC15M val items locally")

            # No cross-rank gather needed: empirically each rank already has
            # the full 10000-item set after iterating val_dataset_yfcc (LitData
            # replicates rather than partitions during direct iteration with
            # the seeded ld.train_test_split). DistributedSampler below then
            # shards the index space across ranks so each rank processes
            # ~len(val_items)/world_size unique samples per epoch — the same
            # number on every rank, which is what eliminates the BARRIER vs
            # GATHER divergence at the validation epoch_end barrier.
            #
            # An all_gather_object here OOMs the GPUs (6GB pickle per rank x 8
            # = 48GB, on top of model+optimizer state) and isn't necessary.

            class _YFCCValMapDataset(torch.utils.data.Dataset):
                def __init__(self, items):
                    self.items = items
                def __len__(self):
                    return len(self.items)
                def __getitem__(self, idx):
                    return self.items[idx]

            val_map_dataset = _YFCCValMapDataset(val_items)
            val_sampler = (
                torch.utils.data.distributed.DistributedSampler(
                    val_map_dataset, shuffle=False, drop_last=False
                )
                if world_size > 1
                else None
            )
            # num_workers=0: val_items is already a Python list of tensors held
            # in the main process. Spawning forked workers (config.num_workers
            # would default to 8) causes each worker to CoW-decow its own ~6GB
            # copy of the list — 8 workers × 8 ranks ≈ 384 GB of host RAM,
            # which exhausts host memory. Validation throughput here
            # is irrelevant (10 batches/rank served from in-memory tensors), so
            # serial loading in the main process is strictly better.
            val_loader_yfcc = torch.utils.data.DataLoader(
                val_map_dataset,
                batch_size=eval_batch_size,
                sampler=val_sampler,
                num_workers=0,
                pin_memory=config.pin_memory,
                collate_fn=skip_indexing_error_collate,
            )
            val_dataloaders.append(val_loader_yfcc) # Add YFCC val loader first
            logger.info(f"Created YFCC15M validation DataLoader (index 0): {len(val_map_dataset)} samples, DistributedSampler={world_size > 1}")

        # --- Create CIFAR Validation Loaders (conditionally) ---
        cifar10_dataloader = None
        cifar100_dataloader = None
        if multi_val:
            logger.info("Creating multiple validation dataloaders (CIFAR10 and CIFAR100)")
            cifar10_dataloader = create_cifar10_validation_dataloader(
                transform=eval_transform,
                batch_size=eval_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                distributed=distributed,
                val_dataset_root=config.val_dataset_root,
                seed=seed
            )
            cifar100_dataloader = create_cifar100_validation_dataloader(
                transform=eval_transform,
                batch_size=eval_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                distributed=distributed,
                val_dataset_root=config.val_dataset_root,
                seed=seed
            )
            if cifar10_dataloader:
                val_dataloaders.append(cifar10_dataloader) # Add CIFAR10 (index 1 or potentially 0)
                logger.info(f"Added CIFAR10 validation DataLoader (index {len(val_dataloaders)-1})")
            if cifar100_dataloader:
                val_dataloaders.append(cifar100_dataloader) # Add CIFAR100 (index 2 or potentially 1)
                logger.info(f"Added CIFAR100 validation DataLoader (index {len(val_dataloaders)-1})")
        else:
            # Original logic: Create just CIFAR10
            logger.info("Creating CIFAR10 validation dataloader.")
            cifar10_dataloader = create_cifar10_validation_dataloader(
                transform=eval_transform,
                batch_size=eval_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                distributed=distributed,
                val_dataset_root=config.val_dataset_root,
                seed=seed
            )
            if cifar10_dataloader:
                 val_dataloaders.append(cifar10_dataloader) # Add CIFAR10 (index 1 or potentially 0)
                 logger.info(f"Added CIFAR10 validation DataLoader (index {len(val_dataloaders)-1})")


        # --- Assign the list of dataloaders ---
        if val_dataloaders:
             dataloaders["val"] = val_dataloaders
             logger.info(f"Created {len(val_dataloaders)} validation dataloader(s) in total.")
        else:
             dataloaders["val"] = None
             logger.warning("No validation dataloaders could be created.")
        
        # --- Test dataloader ---
        dataloaders["test"] = None # Test set not handled in this setup
        logger.info("Created dataloaders for YFCC15M using LitData.")


    else:
        raise ValueError(f"Unsupported dataset_type: {config.dataset_type}")

    return dataloaders

class YFCC15MDataModule(pl.LightningDataModule):
    """Defers create_dataloaders() to setup() so val samplers see an initialized process group.

    Without this, create_dataloaders() runs pre-fit (pre-DDP-init); the CIFAR val helpers
    hit `torch.distributed.is_initialized() == False`, fall back to sampler=None, and every
    rank iterates the full CIFAR test set — 4x redundant eval on 4 GPUs.
    """

    def __init__(self, config, seed: int = 42, distributed: bool = True, multi_val: bool = False):
        super().__init__()
        self._dl_config = config
        self._dl_seed = seed
        self._dl_distributed = distributed
        self._dl_multi_val = multi_val
        self._loaders: Optional[Dict[str, object]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._loaders is not None:
            return
        self._loaders = create_dataloaders(
            config=self._dl_config,
            seed=self._dl_seed,
            distributed=self._dl_distributed,
            multi_val=self._dl_multi_val,
        )

    def train_dataloader(self):
        return self._loaders.get("train") if self._loaders else None

    def val_dataloader(self):
        return self._loaders.get("val") if self._loaders else None

    def test_dataloader(self):
        return self._loaders.get("test") if self._loaders else None
