"""Predict CheXpert labels for one chest X-ray image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

try:
    from .labels import DEFAULT_THRESHOLD
    from .models import build_model
    from .utils import build_transforms, get_device
except ImportError:
    from labels import DEFAULT_THRESHOLD
    from models import build_model
    from utils import build_transforms, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CheXpert labels for one image")
    parser.add_argument("--checkpoint", type=str, default="backend/models/chexpert_multilabel/best_model.pth")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of readable text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    labels = checkpoint["labels"]
    img_size = int(checkpoint.get("img_size", 224))
    model_name = checkpoint.get("model_name", "efficientnet_b0")

    model = build_model(model_name, num_labels=len(labels), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(img_size, train=False)
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        image = transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(image)).squeeze(0).cpu().numpy()

    results = [
        {"label": label, "probability": float(prob), "positive": bool(prob >= args.threshold)}
        for label, prob in zip(labels, probs)
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"Image: {image_path}")
        print(f"Threshold: {args.threshold}")
        for item in results:
            mark = "YES" if item["positive"] else "   "
            print(f"{mark} {item['label']:28s} {item['probability']:.4f}")


if __name__ == "__main__":
    main()
