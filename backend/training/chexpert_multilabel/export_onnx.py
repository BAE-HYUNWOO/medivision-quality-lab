"""Export trained CheXpert model to ONNX for later deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .models import build_model
    from .utils import get_device
except ImportError:
    from models import build_model
    from utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CheXpert model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="backend/models/chexpert_multilabel/best_model.pth")
    parser.add_argument("--output", type=str, default="backend/models/chexpert_multilabel/chexpert_multilabel.onnx")
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

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {output}")


if __name__ == "__main__":
    main()
