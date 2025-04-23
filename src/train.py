"""
Docstring
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from . import get_data_paths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = None


class DRDataSet(Dataset):
    """
    Docstring
    """

    def __init__(self, data_dict, split, task, transform):
        self.image_paths = data_dict[split][task]["images"]
        self.mask_paths = data_dict[split][task]["masks"]
        self.transform = transform

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype("uint8")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

    def __len__(self):
        return len(self.image_paths)


data_paths = get_data_paths("training")

transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(p=0.3),
        A.VerticalFlip(p=0.6),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.2),
        A.Rotate(
            p=0.5,
            limit=(-45, 45),
            rotate_method="largest_box",
            crop_border=False,
        ),
        A.MultiplicativeNoise(
            multiplier=(0.95, 1.05), per_channel=False, elementwise=False, p=1.0
        ),
    ],
    additional_targets={"mask": "mask"},
)

train_dataset = DRDataSet(
    data_dict=data_paths[DATASET], split="train", task="EX", transform=transforms
)
val_dataset = DRDataSet(
    data_dict=data_paths[DATASET], split="val", task="EX", transform=transforms
)
test_dataset = DRDataSet(
    data_dict=data_paths[DATASET], split="test", task="EX", transform=transforms
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
