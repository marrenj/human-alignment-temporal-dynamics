import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext
from tqdm import tqdm

from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset, ThingsFMRIDataset
from temporal_perturbations.data.nod_dataset import NodDataset


def extract_embeddings(model, dataset_name, img_dir, annotations_file, batch_size, num_workers, max_images_per_category, device, logger=None):
    """
    Run a dataset through the model and collect image embeddings.

    Args:
        model: CLIPHBA model instance (already initialized with DoRA layers).
        dataset_name (str): Either 'things' or 'nod'.
        config (dict): Inference configuration containing dataset paths and loader params.
        device (torch.device): Device to perform inference on.
        logger (logging.Logger, optional): Logger for progress messages.

    Returns:
        dict: {
            "image_names": list[str],
            "embeddings": torch.Tensor [N, D],
            "targets" or "categories": labels for each image (dataset-specific)
        }
    """
    if dataset_name not in {"things", "nod"}:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected 'things' or 'nod'.")

    if img_dir is None:
        raise ValueError("img_dir must be provided for embedding extraction.")

    # Build dataset
    import pandas as pd
    if dataset_name == "things" and "concept" not in pd.read_csv(annotations_file, nrows=1).columns:
        dataset = ThingsBehavioralDataset(
            img_annotations_file=annotations_file,
            img_dir=img_dir,
        )
    elif dataset_name == "things" and "concept" in pd.read_csv(annotations_file, nrows=1).columns:
        dataset = ThingsFMRIDataset(
            img_annotations_file=annotations_file,
            img_dir=img_dir,
        )
    elif dataset_name == "nod":
        dataset = NodDataset(
            category_index_file=annotations_file,
            img_dir=img_dir,
            max_images_per_category=max_images_per_category,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected 'things' or 'nod'.")

    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if logger:
        logger.info(f"Extracting embeddings for {len(dataset)} samples from {dataset_name}.")

    all_names = []
    all_embeddings = []
    categories = []

    use_autocast = device.type == "cuda"
    autocast_ctx = torch.amp.autocast if use_autocast else nullcontext

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        progress = tqdm(dataloader, desc=f"Extracting {dataset_name} embeddings", total=len(dataloader))
        for batch in progress:
            if dataset_name == "things" and "concept" not in pd.read_csv(annotations_file, nrows=1).columns:
                image_names, images, _ = batch
            else:
                image_names, images, batch_categories = batch
                categories.extend(batch_categories)

            all_names.extend(list(image_names))
            images = images.to(device, non_blocking=non_blocking)

            with autocast_ctx(device_type="cuda") if use_autocast else autocast_ctx():
                img_embs = model(images)

            all_embeddings.append(img_embs.cpu())

    outputs = {
        "image_names": all_names,
        "embeddings": torch.cat(all_embeddings, dim=0),
        "categories": categories,
    }

    return outputs