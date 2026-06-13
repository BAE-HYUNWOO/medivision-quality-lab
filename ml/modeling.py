from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def get_gradcam_target_layer(model: nn.Module, model_name: str):
    model_name = model_name.lower()
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "mobilenet_v3_small":
        return model.features[-1]
    raise ValueError(f"Unsupported model_name for Grad-CAM: {model_name}")
