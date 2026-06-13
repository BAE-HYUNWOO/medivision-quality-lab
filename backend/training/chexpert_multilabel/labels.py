"""CheXpert label definitions and label conversion helpers."""

from __future__ import annotations

from typing import Iterable

# CheXpert's standard 14 observation labels.
CHEXPERT_LABELS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Default threshold for converting sigmoid probabilities into positive/negative labels.
DEFAULT_THRESHOLD = 0.50

# CheXpert contains uncertain labels marked as -1. There is no single perfect policy.
# zeros: uncertain -> 0. Good first baseline.
# ones: uncertain -> 1. Sometimes useful for selected findings.
# ignore: uncertain/missing labels are ignored in loss computation.
UNCERTAIN_POLICIES = {"zeros", "ones", "ignore"}


def validate_labels(labels: Iterable[str]) -> list[str]:
    labels = list(labels)
    unknown = [label for label in labels if label not in CHEXPERT_LABELS]
    if unknown:
        raise ValueError(f"Unknown CheXpert labels: {unknown}")
    return labels
