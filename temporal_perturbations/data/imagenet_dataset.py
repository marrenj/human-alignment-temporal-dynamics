"""
ImageNet dataset for from-scratch classification training.
 
Expects the standard ImageNet folder structure:
 
    img_dir/
        train/
            n01234567/   <- synset id folders
                img.JPEG
                ...
        val/
            n01234567/
                ...
 
Returns (image_name, image, class_label) tuples to match the rest of the
training pipeline, where class_label is an integer in [0, num_classes).
"""

from torchvision import transforms, datasets
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ImagenetDataset(Dataset):
    """
    Full ImageNet dataset using standard ImageFolder structure.
 
    Args:
        img_dir:  Root directory containing ``train/`` and ``val/`` sub-folders.
        split:    ``'train'`` or ``'val'``.
        img_size: Spatial resolution to resize/crop images to (default 224).
 
    Returns per item:
        (image_name, image_tensor, class_label)
        image_name  – filename (str)
        image_tensor – float32 CHW tensor, normalised with ImageNet stats
        class_label  – integer class index in [0, num_classes)
    """

    def __init__(self, img_dir: str, split: str = 'train', img_size: int = 224):
        split_dir = Path(img_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"ImageNet {split} split not found at: {split_dir}"
            )

        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        # ImageFolder enumerates (path, class_idx) pairs and builds class maps
        self._folder = datasets.ImageFolder(root=str(split_dir))

        print(
            f"ImageNet {split}: {len(self._folder):,} images, "
            f"{len(self._folder.classes)} classes"
        )

    def __len__(self) -> int:
        return len(self._folder)

    def __getitem__(self, index: int):
        path, label  = self._folder.samples[index]
        image_name = Path(path).name
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image_name, image, label

    @property
    def num_classes(self) -> int:
        return len(self._folder.classes)

    @property
    def class_to_idx(self) -> dict:
        return self._folder.class_to_idx