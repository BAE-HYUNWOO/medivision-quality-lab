"""Check whether CheXpert CSV/image paths are correctly placed."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .labels import CHEXPERT_LABELS
    from .utils import resolve_image_path
except ImportError:
    from labels import CHEXPERT_LABELS
    from utils import resolve_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CheXpert dataset placement")
    parser.add_argument("--data-root", type=str, default="data/chexpert")
    parser.add_argument("--train-csv", type=str, default="data/chexpert/train.csv")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.train_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    missing = [label for label in CHEXPERT_LABELS if label not in df.columns]
    if missing:
        print("Missing label columns:", missing)
    else:
        print("All 14 CheXpert label columns found.")

    if "Frontal/Lateral" in df.columns:
        print("Frontal/Lateral counts:")
        print(df["Frontal/Lateral"].value_counts(dropna=False).to_string())

    print("\nPositive label counts where value == 1:")
    for label in CHEXPERT_LABELS:
        if label in df.columns:
            print(f"  {label:28s}: {int((df[label] == 1).sum()):,}")

    print("\nTesting image path resolution:")
    for i in range(min(args.limit, len(df))):
        p = resolve_image_path(args.data_root, str(df.iloc[i]["Path"]))
        print(f"  OK: {p}")

    print("\nDataset placement looks OK.")


if __name__ == "__main__":
    main()
