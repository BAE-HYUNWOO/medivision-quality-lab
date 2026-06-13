from __future__ import annotations

from typing import Dict, Tuple

from medmnist import INFO
import medmnist
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD


def _get_dataset_class(data_flag: str):
    if data_flag not in INFO:
        raise ValueError(f"Unknown MedMNIST data_flag: {data_flag}. Available: {list(INFO.keys())}")
    class_name = INFO[data_flag]["python_class"]
    return getattr(medmnist, class_name)


def get_class_names(data_flag: str) -> list[str]:
    info = INFO[data_flag]
    label_map = info.get("label", {})
    # MedMNIST label keys are often strings: {"0": "normal", "1": "pneumonia"}
    if label_map:
        return [label_map[str(i)] for i in range(len(label_map))]
    return [str(i) for i in range(int(info["n_classes"]))]


def get_num_classes(data_flag: str) -> int:
    return int(INFO[data_flag]["n_classes"])


def build_transforms(size: int = 224) -> Dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "val": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "test": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    }


def get_datasets(data_flag: str = "pneumoniamnist", size: int = 224, download: bool = True):
    dataset_class = _get_dataset_class(data_flag)
    tfms = build_transforms(size)
    datasets = {
        split: dataset_class(split=split, transform=tfms[split], download=download, size=size)
        for split in ["train", "val", "test"]
    }
    return datasets


def get_dataloaders(
    data_flag: str = "pneumoniamnist",
    size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    datasets = get_datasets(data_flag=data_flag, size=size, download=download)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    meta = {
        "class_names": get_class_names(data_flag),
        "num_classes": get_num_classes(data_flag),
        "datasets": datasets,
    }
    return loaders, meta
