import time
import torch
from model_layers import ObliviousNATNet
from evals import test_models
from data import load_data
from config import parse_config

def test_obnat():
    config = parse_config()
    filename = f"{config.dataset}_train_test_val.pt"
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.cache_dir, filename)

    best_params = {  
    "lr": 0.003148121658590051,
    "weight_decay": 0.001254566975334968,
    "batch_size": 4096,
    "depth": 10,
    "n_trees": 152,
    "dropout": 0.001172966795034678,
    "label_smoothing": 0.07605117024796712
    }
    model = ObliviousNATNet(
        input_dim=X_train.shape[1],
        output_dim=7,
        depth=best_params["depth"],
        n_trees=best_params["n_trees"],
        dropout=best_params["dropout"],
    )
    state = torch.load("./runs/covertype_2026-08-11/covertype_2026-08-11_10-40-22/model_weights/optuna_best_covertype.pt", weights_only=True)
    model.load_state_dict(state["best_state"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)


    with torch.inference_mode():
        _ = model(X_test_dev[:256])

    t0 = time.perf_counter()
    results = test_models(model, X_test_dev, y_test_dev)
    infer_time_obnat = time.perf_counter() - t0

    print(f"ObNat — acc: {results['test_acc']:.4f}  auc: {results['test_auc']:.4f}  f1: {results['test_f1']:.4f}")
    print(f"Inference time: {infer_time_obnat:.4f}s  (n={len(X_test_dev)})")


if __name__=="__main__":
    test_obnat()