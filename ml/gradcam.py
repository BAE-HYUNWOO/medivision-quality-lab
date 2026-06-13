from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import IMAGENET_MEAN, IMAGENET_STD
from .modeling import get_gradcam_target_layer
from .predict import build_infer_transform, load_model


def _pil_to_rgb_float(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("RGB").resize((size, size))
    arr = np.array(image).astype(np.float32) / 255.0
    return arr


def generate_gradcam_for_image(
    model,
    model_name: str,
    image: Image.Image,
    size: int,
    device: torch.device,
    target_index: int | None = None,
) -> np.ndarray:
    transform = build_infer_transform(size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    if target_index is None:
        target_index = pred_idx

    target_layers = [get_gradcam_target_layer(model, model_name)]
    targets = [ClassifierOutputTarget(target_index)]

    # reshape_transform is not needed for CNN models used here.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = _pil_to_rgb_float(image, size)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization


def save_gradcam_overlay(model_path: str | Path, image_path: str | Path, output_path: str | Path, cpu: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model, class_names, model_name, size = load_model(model_path, device)
    image = Image.open(image_path).convert("RGB")
    overlay = generate_gradcam_for_image(model, model_name, image, size, device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path
