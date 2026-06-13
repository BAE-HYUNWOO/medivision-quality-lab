from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.core.model_loader import ModelBundle
from app.services.inference_service import build_transform, read_image_bytes, predict_image_bytes


def get_target_layer(bundle: ModelBundle):
    model_name = bundle.model_name.lower()
    if model_name == "resnet18":
        return bundle.model.layer4[-1]
    if model_name == "efficientnet_b0":
        return bundle.model.features[-1]
    if model_name == "mobilenet_v3_small":
        return bundle.model.features[-1]
    raise ValueError(f"Unsupported model for Grad-CAM: {bundle.model_name}")


def image_to_rgb_float(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("RGB").resize((size, size))
    return np.array(image).astype(np.float32) / 255.0


def png_base64_from_array(rgb_array: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Could not encode Grad-CAM image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def predict_with_gradcam(bundle: ModelBundle, image_bytes: bytes) -> dict:
    bundle.load()
    prediction = predict_image_bytes(bundle, image_bytes)
    image = read_image_bytes(image_bytes)
    transform = build_transform(bundle.size)
    x = transform(image).unsqueeze(0).to(bundle.device)

    target_index = prediction["predicted_index"]
    target_layers = [get_target_layer(bundle)]
    targets = [ClassifierOutputTarget(target_index)]

    with GradCAM(model=bundle.model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=x, targets=targets)[0]

    rgb_img = image_to_rgb_float(image, bundle.size)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    prediction["gradcam_png_base64"] = png_base64_from_array(overlay)
    return prediction
