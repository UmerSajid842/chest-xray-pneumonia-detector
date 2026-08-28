"""Dataset transforms and dataloaders for the chest X-ray classifier."""
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def build_transforms(image_size: int = IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return training and evaluation transforms without reading any files."""
    train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    evaluation = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train, evaluation


def build_dataloaders(data_root: Path, batch_size: int = 32, num_workers: int = 0):
    """Build ImageFolder loaders from train/val/test directories."""
    train_tfms, eval_tfms = build_transforms()
    splits = {}
    for split, tfms in (("train", train_tfms), ("val", eval_tfms), ("test", eval_tfms)):
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing dataset split: {split_dir}")
        splits[split] = datasets.ImageFolder(split_dir, transform=tfms)
    return {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
        for split, dataset in splits.items()
    }
