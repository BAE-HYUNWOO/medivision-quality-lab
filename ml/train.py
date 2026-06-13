from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import IMAGENET_MEAN, IMAGENET_STD
from .dataset import get_dataloaders
from .modeling import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def labels_to_1d(labels: torch.Tensor) -> torch.Tensor:
    # MedMNIST labels are often shape [B, 1]. CrossEntropyLoss needs [B].
    if labels.ndim > 1:
        labels = labels.squeeze()
    return labels.long()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    dataloaders, meta = get_dataloaders(
        data_flag=args.data_flag,
        size=args.size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )

    class_names = meta["class_names"]
    num_classes = meta["num_classes"]
    print(f"Dataset: {args.data_flag}")
    print(f"Classes ({num_classes}): {class_names}")

    model = build_model(args.model_name, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_correct = 0
            total = 0

            pbar = tqdm(dataloaders[phase], desc=phase)
            for images, labels in pbar:
                images = images.to(device)
                labels = labels_to_1d(labels).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    preds = torch.argmax(logits, dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_correct += (preds == labels).sum().item()
                total += batch_size
                pbar.set_postfix(loss=running_loss / max(total, 1), acc=running_correct / max(total, 1))

            epoch_loss = running_loss / total
            epoch_acc = running_correct / total
            print(f"{phase}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    print(f"\nBest val acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "data_flag": args.data_flag,
            "size": args.size,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
            "best_val_acc": best_val_acc,
        },
        output_path,
    )
    print(f"Saved checkpoint: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MedMNIST classifier")
    parser.add_argument("--data-flag", default="pneumoniamnist")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--model-name", default="resnet18", choices=["resnet18", "efficientnet_b0", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-path", default="models/pneumonia_resnet18_224.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
