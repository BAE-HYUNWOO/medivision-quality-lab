"""Evaluate a trained CheXpert multi-label model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset_chexpert import CheXpertDataset
    from .metrics import compute_multilabel_metrics
    from .models import build_model
    from .utils import build_transforms, get_device
except ImportError:
    from dataset_chexpert import CheXpertDataset
    from metrics import compute_multilabel_metrics
    from models import build_model
    from utils import build_transforms, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CheXpert multi-label classifier")
    parser.add_argument("--checkpoint", type=str, default="backend/models/chexpert_multilabel/best_model.pth")
    parser.add_argument("--data-root", type=str, default="data/chexpert")
    parser.add_argument("--csv", type=str, default="data/chexpert/valid.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", type=str, default="backend/models/chexpert_multilabel/eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    labels = checkpoint["labels"]
    img_size = int(checkpoint.get("img_size", 224))
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    uncertain_policy = checkpoint.get("uncertain_policy", "zeros")
    frontal_only = bool(checkpoint.get("frontal_only", True))

    dataset = CheXpertDataset(
        csv_path=args.csv,
        data_root=args.data_root,
        labels=labels,
        transform=build_transforms(img_size, train=False),
        frontal_only=frontal_only,
        uncertain_policy=uncertain_policy,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_name, num_labels=len(labels), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)
            total_loss += float(loss.item()) * images.size(0)
            total_items += images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_prob, labels)
    metrics["loss"] = total_loss / max(total_items, 1)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
