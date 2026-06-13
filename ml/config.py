from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_flag: str = "pneumoniamnist"
    size: int = 224
    model_name: str = "resnet18"
    batch_size: int = 32
    epochs: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    output_path: Path = Path("models/pneumonia_resnet18_224.pt")
    seed: int = 42


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
