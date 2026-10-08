import torch
import numpy as np
import random
import os


def save_random_states(optimizer, epoch, random_state_path, dataloader_generator, 
logger=None, scheduler=None):
    """
    Save all random states and optimizer state for 100% reproducibility, 
    and to be used for resuming training.
    
    Args:
        optimizer: The optimizer whose state to save
        epoch: Current epoch number
        random_state_path: Directory to save the checkpoint
        logger: Optional logger for logging messages
        scheduler: Optional LR scheduler whose state to save

    Returns:
        None. Writes a checkpoint file containing RNG and optimizer state.
    """
    log = logger.info if logger else print
    
    # Create checkpoint with all random states
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'dataloader_generator_state': dataloader_generator.get_state(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save CUDA random states for all GPUs if available
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    
    # Save the checkpoint
    os.makedirs(random_state_path, exist_ok=True)
    checkpoint_file = os.path.join(random_state_path, f"epoch{epoch}_random_states.pth")
    torch.save(checkpoint, checkpoint_file)
    log(f"Random states saved: {checkpoint_file}")