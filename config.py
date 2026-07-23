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
    ckpt:Path = Path("./checkpoints/checkpoint_obnat.pt")
    results:Path = Path("./results/Oblivious_nat_results.json")
    n_classes:int =7
    depth:int = 8
    n_trees:int = 144

def parse_config() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="ObnatNet")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weigt decay")
    parser.add_argument("--patience", type=int, default=None, help="Early stop after")
    parser.add_argument("--lr", type=float, default=None, help="Model Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Size of each batch processed")
    parser.add_argument("--cache_dir", default=None, help="Directory of your file location")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path for last file")
    parser.add_argument("--results", default=None, help="saves json results for the training")
    parser.add_argument("--n_classes",default=None, help="Number of classes of dataset")
    parser.add_argument("--depth", type=int, default=None, help="Tree depth")
    parser.add_argument("--n_trees", type=int, default=None, help="Number of trees per layer")
    args = parser.parse_args()

    config = TrainingConfig()
    overrides = {key:value for key,value in vars(args).items() if value is not None}

    return replace(config, **overrides)



