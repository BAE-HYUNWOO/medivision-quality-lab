"""Utility functions for CheXpert multi-label training."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Build light augmentation suitable for chest X-ray classification.

    The model uses ImageNet-pretrained backbones, so grayscale X-rays are converted
    to 3-channel RGB and normalized with ImageNet statistics.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_image_path(data_root: str | Path, csv_path_value: str) -> Path:
    """Resolve CheXpert image paths robustly.

    CheXpert CSVs often store paths like:
        CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg

    If the user moves train.csv, valid.csv, train/, valid/ into data/chexpert,
    this helper strips the CheXpert folder prefix and still finds the image.
    """
    data_root = Path(data_root)
    raw_path = Path(str(csv_path_value).replace("\\", "/"))

    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)

    candidates.append(data_root / raw_path)

    parts = list(raw_path.parts)
    for idx, part in enumerate(parts):
        if part in {"train", "valid", "test"}:
            candidates.append(data_root / Path(*parts[idx:]))
            break

    # Some downloads are left as data/chexpert/CheXpert-v1.0-small/...
    candidates.append(data_root / "CheXpert-v1.0-small" / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find image file for CSV path:\n"
        f"  {csv_path_value}\nTried:\n  - {joined}\n\n"
        "Fix by putting train.csv, valid.csv, train/, valid/ under data/chexpert/ "
        "or pass --data-root to the folder that contains the CheXpert images."
    )
