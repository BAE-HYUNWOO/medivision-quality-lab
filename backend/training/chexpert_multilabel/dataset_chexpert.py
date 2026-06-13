"""PyTorch Dataset for CheXpert multi-label classification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from .labels import CHEXPERT_LABELS, UNCERTAIN_POLICIES, validate_labels
    from .utils import resolve_image_path
except ImportError:  # Allows direct script execution from this folder.
    from labels import CHEXPERT_LABELS, UNCERTAIN_POLICIES, validate_labels
    from utils import resolve_image_path


class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        labels: list[str] | None = None,
        transform: Callable | None = None,
        frontal_only: bool = True,
        uncertain_policy: str = "zeros",
        max_samples: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.labels = validate_labels(labels or CHEXPERT_LABELS)
        self.transform = transform
        self.uncertain_policy = uncertain_policy

        if uncertain_policy not in UNCERTAIN_POLICIES:
            raise ValueError(f"uncertain_policy must be one of {UNCERTAIN_POLICIES}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if "Path" not in df.columns:
            raise ValueError(f"CSV must contain a Path column. Found columns: {list(df.columns)}")

        missing_labels = [label for label in self.labels if label not in df.columns]
        if missing_labels:
            raise ValueError(f"CSV is missing label columns: {missing_labels}")

        if frontal_only and "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"].astype(str).str.lower() == "frontal"].copy()

        # Deterministic subset for quick tests.
        if max_samples is not None and max_samples > 0:
            df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _convert_labels(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        raw = row[self.labels].astype("float32").to_numpy()

        # mask=1 means the label participates in loss. For zeros/ones policy,
        # all labels participate. For ignore, uncertain/missing labels do not.
        mask = np.ones_like(raw, dtype=np.float32)

        if self.uncertain_policy == "zeros":
            # missing and uncertain labels become negative.
            raw = np.nan_to_num(raw, nan=0.0)
            raw[raw == -1.0] = 0.0
        elif self.uncertain_policy == "ones":
            raw = np.nan_to_num(raw, nan=0.0)
            raw[raw == -1.0] = 1.0
        elif self.uncertain_policy == "ignore":
            mask[np.isnan(raw)] = 0.0
            mask[raw == -1.0] = 0.0
            raw = np.nan_to_num(raw, nan=0.0)
            raw[raw == -1.0] = 0.0

        raw = np.clip(raw, 0.0, 1.0)
        return torch.tensor(raw, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        image_path = resolve_image_path(self.data_root, str(row["Path"]))

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels, mask = self._convert_labels(row)

        return {
            "image": image,
            "labels": labels,
            "mask": mask,
            "path": str(image_path),
        }
