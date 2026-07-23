from dataclasses import dataclass, replace
from pathlib import Path
import argparse

@dataclass
class TrainingConfig:
    epochs: int = 150
    weight_decay: float = 1e-4
    patience: int = 15
    lr : float = 3e-3
    batch_size: int = 4096
    cache_dir: Path = Path("./dataset_cache/covertype/covertype_train_test_val.pt")


def parse_config() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="ObnatNet")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weigt decay")
    parser.add_argument("--patience", type=int, default=None, help="Early stop after")
    parser.add_argument("--lr", type=float, default=None, help="Model Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Size of each batch processed")
    parser.add_argument("--cache_dir", default=None, help="Directory of your file location")
    args = parser.parse_args()

    config = TrainingConfig()
    overrides = {key:value for key,value in vars(args).items() if value is not None}

    return replace(config, **overrides)



