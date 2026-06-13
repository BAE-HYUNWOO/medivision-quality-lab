from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models


class ModelLoadError(RuntimeError):
    pass


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ModelLoadError(f"Unsupported model name: {model_name}")


class ModelBundle:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names: list[str] = []
        self.model_name = "resnet18"
        self.size = 224
        self._loaded = False

    def available(self) -> bool:
        return self.model_path.exists() and self.model_path.is_file()

    def load(self):
        if self._loaded:
            return self
        if not self.available():
            raise ModelLoadError(f"Model checkpoint not found: {self.model_path}")

        checkpoint: dict[str, Any] = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        self.model_name = checkpoint.get("model_name", "resnet18")
        self.size = int(checkpoint.get("size", 224))
        self.model = build_model(self.model_name, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return self
