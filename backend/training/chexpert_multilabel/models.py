"""Model builders for CheXpert multi-label classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_labels: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_labels)
        return model

    if model_name == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_labels)
        return model

    if model_name == "densenet121":
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
        except AttributeError:
            model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_labels)
        return model

    raise ValueError(
        f"Unsupported model '{model_name}'. Use one of: efficientnet_b0, resnet50, densenet121"
    )
