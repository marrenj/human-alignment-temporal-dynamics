

def count_trainable_parameters(model):
    """Return the number of trainable parameters in ``model``.

    Args:
        model: Any object exposing a ``parameters`` iterator yielding tensors.

    Returns:
        int: Total parameter count for tensors with ``requires_grad=True``.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)