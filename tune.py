import csv
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from pathlib import Path
from optuna_util import GlobalBest
from config import TrainingConfig, parse_config
from optuna_objective import objective
from data import load_data
from training_utils import atomic_save_json
from model_layers import ObliviousNATNet
from evals import test_models


def log_trial_callback(study, trial, config:TrainingConfig):
    row = {
        "trial_number":  trial.number,
        "state":         trial.state.name,
        "value_val_acc": trial.value,
        "test_acc":      trial.user_attrs.get("test_acc"),
        "test_auc":      trial.user_attrs.get("test_auc"),
        "params_count":  trial.user_attrs.get("params"),
        **trial.params,  # n_trees, lr, weight_decay, batch_size, dropout, label_smoothing
    }
    config.trial_log.mkdir(parents=True, exist_ok=True)
    log_path = Path(config.trial_log / f'{config.dataset}_result.csv')
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def optuna_study(input_dim, X_train, y_train, X_val, y_val, X_test, y_test, device, config:TrainingConfig):
    global_best = GlobalBest()
    
    study = optuna.create_study(
        study_name=config.study_name,
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=HyperbandPruner(min_resource=5, max_resource=config.epochs, reduction_factor=3)
    )
    study.optimize(
        lambda trial:objective(input_dim, X_train, y_train, X_val, y_val, device, config, trial, global_best),
        n_trials= config.n_trials,
        gc_after_trial=True,
        callbacks=[    
            lambda study, trial: log_trial_callback(
            study,
            trial,
            config,
        )]
    )

    best = study.best_trial

    if global_best.state is None or global_best.config is None:
        raise RuntimeError("Optuna finished without saving a valid best model.")

    # Use the configuration corresponding to the saved global-best state.
    best_config = global_best.config

    best_model = ObliviousNATNet(
        input_dim=input_dim,
        output_dim=best_config.n_classes,
        depth=best_config.depth,
        n_trees=best_config.n_trees,
        dropout=best_config.dropout,
    ).to(device)

    best_model.load_state_dict(global_best.state)
    best_model.eval()

    # Evaluate the test set exactly once.
    test_results = test_models(
        best_model,
        X_test,
        y_test,
    )

    result = {
        "best_val_acc": global_best.val_acc,
        "best_params": {
            "lr": best_config.lr,
            "weight_decay": best_config.weight_decay,
            "batch_size": best_config.batch_size,
            "depth": best_config.depth,
            "n_trees": best_config.n_trees,
            "dropout": best_config.dropout,
            "label_smoothing": best_config.label_smoothing,
        },
        "test_acc": test_results["test_acc"],
        "test_auc": test_results["test_auc"],
        "test_f1": test_results["test_f1"],
        "model_params": best.user_attrs.get("params"),
        "best_trial": global_best.trial_number,
        "best_epoch": global_best.best_epoch,
        "n_trials": len(study.trials),
    }
    result_file = f'{config.dataset}_result.json'
    atomic_save_json(result, Path(config.results/result_file))
    config.model_weights.mkdir(parents=True, exist_ok=True)
    torch.save({"best_state": global_best.state},config.model_weights / f"optuna_best_{config.dataset}.pt")

    log_results = Path(config.trial_log / f'{config.dataset}.csv')
    full_log_path = log_results.with_name(f"{log_results.stem}_full.csv")
    study.trials_dataframe().to_csv(full_log_path, index=False)

if __name__=="__main__":
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = 64
    config = parse_config()
    filename = f"{config.dataset}_train_test_val.pt"
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.cache_dir, filename)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print()
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    input_dim = X_train.shape[1]

    optuna_study(input_dim, X_train, y_train, X_val, y_val, X_test, y_test, device, config)

