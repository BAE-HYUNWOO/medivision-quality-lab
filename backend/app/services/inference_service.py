from __future__ import annotations

from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.core.model_loader import ModelBundle

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DISCLAIMER = "Research demo only. Not for clinical diagnosis or treatment decisions."


def read_image_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def build_transform(size: int):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict_image_bytes(bundle: ModelBundle, image_bytes: bytes) -> dict:
    bundle.load()
    image = read_image_bytes(image_bytes)
    transform = build_transform(bundle.size)
    x = transform(image).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return {
        "predicted_index": pred_idx,
        "predicted_label": bundle.class_names[pred_idx],
        "confidence": confidence,
        "probabilities": {bundle.class_names[i]: float(probs[i].item()) for i in range(len(bundle.class_names))},
        "disclaimer": DISCLAIMER,
    }
