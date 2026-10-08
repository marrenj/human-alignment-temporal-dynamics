import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

from temporal_perturbations.data.eval_transforms import THINGS_LEGACY

# The historical transform for these stimuli: squash to 224x224, THINGS image-set
# mean/std. Kept as the default for every consumer so CLIP-HBA results are unchanged.
DEFAULT_THINGS_TRANSFORM = THINGS_LEGACY.build(224)


def resolve_things_image_path(img_dir, image_name):
    """Resolve a THINGS image name against either on-disk layout.

    Two trees are in use, differing only in nesting:

      * FLAT   -- ``<img_dir>/<image_name>``
      * NESTED -- ``<img_dir>/<concept>/<image_name>``

    where the concept is the filename minus its trailing image-number token
    (``water_gun_01b.jpg`` -> ``water_gun/``, ``aardvark_01b.jpg`` -> ``aardvark/``)
    -- the same grouping ``ThingsFMRIDataset`` gets from its explicit ``concept``
    column.

    Flat is tried first so any tree that already worked keeps its exact previous
    behaviour and pays only one extra ``exists`` check. Falling back rather than
    detecting the layout once also tolerates a partially-nested tree. If neither
    exists we raise naming both candidates, which is far easier to act on than
    the bare FileNotFoundError PIL raises for whichever single path was built.
    """
    flat = os.path.join(img_dir, image_name)
    if os.path.exists(flat):
        return flat

    concept = '_'.join(image_name.split('_')[:-1])
    nested = os.path.join(img_dir, concept, image_name)
    if os.path.exists(nested):
        return nested

    raise FileNotFoundError(
        f"THINGS image {image_name!r} not found under {img_dir!r}. Tried:\n"
        f"  flat:   {flat}\n"
        f"  nested: {nested}"
    )


class ThingsBehavioralDataset(Dataset):
    def __init__(self, img_annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        # `transform=None` keeps the historical THINGS-statistics squash-resize, which
        # every CLIP-HBA number in this repo was produced under -- do not change the
        # default. Cross-architecture callers pass the recipe their checkpoints were
        # trained under instead; see data/eval_transforms.py.
        self.transform = transform or DEFAULT_THINGS_TRANSFORM

        # Load and filter annotations based on the 'set' column
        self.annotations = pd.read_csv(img_annotations_file, index_col=0)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        img_path = resolve_things_image_path(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)


        targets = torch.tensor(self.annotations.iloc[index, 1:].values.astype('float32'))

        return image_name, image, targets


class ThingsFMRIDataset(Dataset):
    def __init__(self, img_annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform or DEFAULT_THINGS_TRANSFORM  # see ThingsBehavioralDataset

        # Load the full CSV
        self.stimuli_metadata = pd.read_csv(img_annotations_file, index_col=0)

    def __len__(self):
        return len(self.stimuli_metadata)

    def __getitem__(self, index):
        # Get the row at the specified index
        row = self.stimuli_metadata.iloc[index]

        # Extract the concept and image name from the row
        concept = row['concept']
        image_name = row['stimulus']

        # Build the image path - images are organized in concept subdirectories
        # Path structure: img_dir/concept/image_name
        img_path = os.path.join(self.img_dir, concept, image_name)

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image_name, image, concept