from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def export_metadata(args):
    checkpoint = torch.load(args.model_path, map_location="cpu")
    metadata = {
        "model_name": checkpoint.get("model_name"),
        "data_flag": checkpoint.get("data_flag"),
        "size": checkpoint.get("size"),
        "class_names": checkpoint.get("class_names"),
        "num_classes": checkpoint.get("num_classes"),
        "best_val_acc": checkpoint.get("best_val_acc"),
        "normalization_mean": checkpoint.get("normalization_mean"),
        "normalization_std": checkpoint.get("normalization_std"),
        "research_only": True,
        "clinical_use": False,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote metadata: {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export model metadata")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-json", default="models/model_metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    export_metadata(parse_args())
