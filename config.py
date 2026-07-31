import os
from dataclasses import dataclass, replace, field
from pathlib import Path
import argparse
import optuna


def last_completed(path:Path, count):
    file = path / f'{count}.runs'/'checkpoint_completed.pt'
    return file.exists()


@dataclass
class TrainingConfig:
    epochs: int = 150
    weight_decay: float = 1e-4
    patience: int = 15
    lr : float = 3e-3
    batch_size: int = 4096
    label_smoothing:float = 0.0
    n_trials:int = 50 
    dropout:float = 0.0

    study_name:str = "Oblivious_nat_Covertype"
    cache_dir: Path = Path("./dataset_cache/covertype")

    run_root: Path = Path("./runs")
    ckpt:Path = field(init=False)
    # full_results:Path= Path("./runs/results/full_results.json")    Need to Fix full_results.json in tune.py 
    results:Path = field(init=False)
    model_weights:Path = field(init=False) 
    # trial_log:Path = Path("./runs/trial_logs/trial_log.csv")   Need to Fix trial_log.csv in tune.py

    n_classes:int =7
    depth:int = 8
    n_trees:int = 144

    checkpoint_save:int = 10

    def __post_init__(self)-> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        count = len(os.listdir(self.run_root))
        run_dir = None
        if count > 0:
            if last_completed(self.run_root, count):
                count +=1
            run_dir = self.run_root/Path(f'{count}.runs')
        else:
            run_dir = self.run_root/f'1.runs'
        run_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt =  run_dir/'checkpoints'
        self.results = run_dir/'results'
        self.model_weights = run_dir/'model_weights'

def parse_config() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="ObnatNet")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weigt decay")
    parser.add_argument("--patience", type=int, default=None, help="Early stop after")
    parser.add_argument("--lr", type=float, default=None, help="Model Learning rate")
    parser.add_argument("--label_smoothing", type=float, default=None, help="Cross Entropy label Smoothing")
    parser.add_argument("--batch_size", type=int, default=None, help="Size of each batch processed")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout Value")
    parser.add_argument("--n_trials", type=int, default=None,help="Number of optuna Trials")
    parser.add_argument("--study_name",  default=None, help="optuna study name")
    parser.add_argument("--run_root", default=None, help="root directory for all the run saves")
    parser.add_argument("--cache_dir", default=None, help="Directory of your file location")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path for last file")
    parser.add_argument("--full_results", default=None, help="all results saved here")
    parser.add_argument("--results", default=None, help="saves json results for the training")
    parser.add_argument("--model_weights", default=None, help="model_weights save location")
    parser.add_argument("--trial_log", default=None, help="Trial log location")
    parser.add_argument("--n_classes",type=int, default=None, help="Number of classes of dataset")
    parser.add_argument("--depth", type=int, default=None, help="Tree depth")
    parser.add_argument("--n_trees", type=int, default=None, help="Number of trees per layer")
    parser.add_argument("--checkpoint_save",type=int, default=None, help="Save last after every 10 epochs")
    args = parser.parse_args()

    path_fields = {
        "cache_dir",
        "ckpt",
        "full_results",
        "results",
        "model_weights",
        "trial_log",
     }

    overrides = {
        key: Path(value) if key in path_fields else value
        for key, value in vars(args).items()
        if value is not None
    }

    return TrainingConfig(**overrides)


def trial_config(trial: optuna.Trial, base_config:TrainingConfig) -> TrainingConfig:
    lr = trial.suggest_float("lr",1e-4,1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay",1e-6,1e-2,log=True)
    batch_size = trial.suggest_categorical("batch_size",[4096])
    depth = trial.suggest_categorical("depth",[4,6,8,10])
    n_trees = trial.suggest_categorical("n_trees",[64,96,112,120,128,136,144,152,192])
    dropout = trial.suggest_float("dropout",0.0,0.3)
    label_smoothing = trial.suggest_float("label_smoothing",0.0,0.1)
    
    return replace(
        base_config,
        lr = lr,
        weight_decay=weight_decay,
        batch_size = batch_size,
        depth = depth,
        n_trees = n_trees,
        dropout = dropout,
        label_smoothing = label_smoothing   
    )
    




