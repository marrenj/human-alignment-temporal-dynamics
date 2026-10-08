import torch
import random
import numpy as np

def seed_everything(seed):
    """
    Seed all major RNG backends (PyTorch CPU/GPU, NumPy, Python's random) and
    force deterministic CuDNN behavior to make experiments reproducible.

    Parameters
    ----------
    seed : int
        The integer seed applied to every supported backend.
    """
    # Set the seed for PyTorch's random number generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set the seed for Python's random number generator
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Ensure that the CuDNN backend is deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False