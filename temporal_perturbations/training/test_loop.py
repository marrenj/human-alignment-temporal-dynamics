import torch
from tqdm import tqdm


def evaluate_model(model, data_loader, device, criterion):
    """Compute mean loss for a model over a dataset.

    Args:
        model (torch.nn.Module): Network to evaluate.
        data_loader (torch.utils.data.DataLoader): Iterable yielding
            ``(_, images, targets)`` batches.
        device (torch.device | str): Device to move inputs/targets onto.
        criterion (Callable): Loss function accepting ``(predictions, targets)``.

    Returns:
        float: Average loss across every sample in ``data_loader.dataset``.
    """
    model.eval()
    total_loss = 0.0

    # Wrap data_loader with tqdm for a progress bar
    with torch.no_grad(), tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating") as progress_bar:
        for _, (_, images, targets) in progress_bar:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(images)

            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            total_loss += loss.item() * images.size(0) 

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss