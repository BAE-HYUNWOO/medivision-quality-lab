"""Train a CheXpert multi-label chest X-ray classifier.

Example quick test:
python backend/training/chexpert_multilabel/train_chexpert.py ^
  --data-root data/chexpert ^
  --train-csv data/chexpert/train.csv ^
  --valid-csv data/chexpert/valid.csv ^
  --epochs 1 --batch-size 16 --max-train-samples 2000 --max-val-samples 500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset_chexpert import CheXpertDataset
    from .labels import CHEXPERT_LABELS
    from .metrics import compute_multilabel_metrics
    from .models import build_model
    from .utils import build_transforms, ensure_dir, get_device, set_seed
except ImportError:
    from dataset_chexpert import CheXpertDataset
    from labels import CHEXPERT_LABELS
    from metrics import compute_multilabel_metrics
    from models import build_model
    from utils import build_transforms, ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CheXpert multi-label classifier")
    parser.add_argument("--data-root", type=str, default="data/chexpert")
    parser.add_argument("--train-csv", type=str, default="data/chexpert/train.csv")
    parser.add_argument("--valid-csv", type=str, default="data/chexpert/valid.csv")
    parser.add_argument("--output-dir", type=str, default="backend/models/chexpert_multilabel")
    parser.add_argument("--checkpoint-dir", type=str, default="backend/checkpoints/chexpert_multilabel")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50", "densenet121"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uncertain-policy", type=str, default="zeros", choices=["zeros", "ones", "ignore"])
    parser.add_argument("--no-frontal-only", action="store_true", help="Use both frontal and lateral images")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet pretrained weights")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for quick debugging")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples for quick debugging")
    return parser.parse_args()


def masked_bce_loss(
    criterion: nn.BCEWithLogitsLoss,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    use_mask: bool,
) -> torch.Tensor:
    if not use_mask:
        return criterion(logits, targets)

    loss_per_label = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = loss_per_label * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    labels: list[str],
    use_mask: bool,
) -> tuple[float, dict]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss = 0.0
    total_items = 0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="valid", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss = masked_bce_loss(criterion, logits, targets, mask, use_mask)

            probs = torch.sigmoid(logits)
            total_loss += float(loss.item()) * images.size(0)
            total_items += images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_prob, labels)
    return total_loss / max(total_items, 1), metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    device = get_device()

    labels = CHEXPERT_LABELS
    frontal_only = not args.no_frontal_only

    print(f"Device: {device}")
    print(f"Labels: {len(labels)}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.img_size}")
    print(f"Uncertain policy: {args.uncertain_policy}")
    print(f"Frontal only: {frontal_only}")

    train_ds = CheXpertDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        labels=labels,
        transform=build_transforms(args.img_size, train=True),
        frontal_only=frontal_only,
        uncertain_policy=args.uncertain_policy,
        max_samples=args.max_train_samples,
    )
    valid_ds = CheXpertDataset(
        csv_path=args.valid_csv,
        data_root=args.data_root,
        labels=labels,
        transform=build_transforms(args.img_size, train=False),
        frontal_only=frontal_only,
        uncertain_policy=args.uncertain_policy,
        max_samples=args.max_val_samples,
    )

    print(f"Train images: {len(train_ds):,}")
    print(f"Valid images: {len(valid_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model, num_labels=len(labels), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    use_mask = args.uncertain_policy == "ignore"
    best_score = -math.inf
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = model(images)
                loss = masked_bce_loss(criterion, logits, targets, mask, use_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
            progress.set_postfix(loss=running_loss / max(seen, 1))

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_loss, val_metrics = run_validation(model, valid_loader, device, labels, use_mask)
        mean_auc = val_metrics["mean_auroc"]
        mean_ap = val_metrics["mean_average_precision"]
        elapsed_min = (time.time() - start) / 60

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_auroc": mean_auc,
            "mean_average_precision": mean_ap,
            "minutes": elapsed_min,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} mean_auroc={mean_auc:.4f} "
            f"mean_AP={mean_ap:.4f} time={elapsed_min:.1f}m"
        )

        # Prefer mean AUROC when available. If unavailable, use negative validation loss.
        score = mean_auc if not math.isnan(mean_auc) else -val_loss
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "labels": labels,
            "img_size": args.img_size,
            "uncertain_policy": args.uncertain_policy,
            "frontal_only": frontal_only,
            "epoch": epoch,
            "metrics": val_metrics,
            "args": vars(args),
        }

        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pth")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"Saved best model to {output_dir / 'best_model.pth'}")

        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print("Training finished.")
    print(f"Best model: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
