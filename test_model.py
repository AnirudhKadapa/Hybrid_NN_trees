import torch
import torch.nn as nn
from evals import chunked_probs
from sklearn.metrics import roc_auc_score, f1_score

@torch.inference_mode()
def test_models(model:nn.Module, X_test:torch.Tensor, y_test:torch.Tensor):
    model.eval()
    
    probs = chunked_probs(model, X_test, y_test)
    probs_cpu = probs.cpu().numpy()
    predictions = probs.argmax(1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    test_acc = float((predictions==y_true).mean())
    test_f1  = float(f1_score(y_true, predictions, average="macro"))
    try:
        test_auc = float(roc_auc_score(
            y_true, probs_cpu, multi_class="ovr", average="macro"))
    except Exception:
        test_auc = 0.0

    results = {
            "model":        "PerTreeObliviousNAT",
            "test_acc":     test_acc,
            "test_auc":     test_auc,
            "test_f1":      test_f1,
        }
    return results