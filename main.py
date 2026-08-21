import time
import torch
from pathlib import Path
from data import load_data
from config import parse_config
from model_layers import ObliviousNATNet
from train import training
from training_utils import atomic_save_json
from test_params import params
from dataclasses import replace
from evals import test_models

def main():
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = 64
    base_config = parse_config()
    best_params = params(base_config.dataset)
    if best_params:
        config = replace(base_config, **best_params)
    else:
        config = base_config

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

    
    model = ObliviousNATNet(input_dim, config.n_classes, config.depth, config.n_trees, config.dropout)

    t_start = time.perf_counter()
    history, best_val_accuracy, best_state = training(model,X_train,X_val,y_train,y_val, device, config)
    elapsed = time.perf_counter()-t_start

    config.model_weights.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        raise RuntimeError(
            "Training ended before producing a valid best model state"
        )
    torch.save(best_state,config.model_weights/f"obnat_{config.dataset}.pt")
    model.load_state_dict(best_state)
    

    model.eval()

    t1 = time.perf_counter()
    test_results = test_models(model, X_test, y_test)
    elapsed1 = time.perf_counter()-t1

    results = {
        "dataset":      config.dataset,
        "n_trees":      config.n_trees,
        "depth":        config.depth,
        "best_val_acc": best_val_accuracy,
        "test_acc":     test_results['test_acc'],
        "test_auc":     test_results['test_auc'],
        "test_f1":      test_results['test_f1'],
        "train_time":   round(elapsed, 3),
        "test_time":    round(elapsed1,3),
        "history":      history,
    }
    file = 'obnat_results.json'
    file_path = config.results/file
    atomic_save_json(results, file_path)

    print("-"*70)
    print(f"Oblivious Nat Net: Results- {config.dataset}")
    print('-'*70)
    print(f"Architecture: Number of Trees:{config.n_trees}, depth:{config.depth}")
    print(f"Best Validation Accuracy: {results['best_val_acc']}")
    print(f"Test Accuracy: {results['test_acc']}")
    print(f"Auc Score: {results['test_auc']}")
    print(f"F1 score: {results['test_f1']}")
    print(f"Total Training time: {results['train_time']}")
    
if __name__=="__main__":
    main()


    




    