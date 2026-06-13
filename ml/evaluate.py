from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from .dataset import get_dataloaders
from .modeling import build_model
from .train import labels_to_1d


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    data_flag = args.data_flag or checkpoint.get("data_flag", "pneumoniamnist")
    size = args.size or checkpoint.get("size", 224)
    model_name = args.model_name or checkpoint.get("model_name", "resnet18")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    loaders, _ = get_dataloaders(
        data_flag=data_flag,
        size=size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loaders["test"], desc="test"):
            images = images.to(device)
            labels = labels_to_1d(labels).cpu().numpy()
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    if num_classes == 2:
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            print("ROC AUC:", auc)
        except Exception as exc:
            print("ROC AUC failed:", exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a MedMNIST classifier")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-flag", default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
