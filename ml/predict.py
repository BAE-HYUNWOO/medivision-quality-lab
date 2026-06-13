from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD
from .modeling import build_model


def build_infer_transform(size: int):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(model_path: str | Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "resnet18")
    size = checkpoint.get("size", 224)

    model = build_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, model_name, size


def predict_image(model, class_names, image_path, size, device):
    transform = build_infer_transform(size)
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
    return {
        "predicted_index": pred_idx,
        "predicted_label": class_names[pred_idx],
        "confidence": confidence,
        "probabilities": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, class_names, model_name, size = load_model(args.model_path, device)
    result = predict_image(model, class_names, args.image_path, size, device)
    print(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a single medical image")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
