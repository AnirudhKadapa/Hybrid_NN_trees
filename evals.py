import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

@torch.inference_mode()
def chunked_probs(model:nn.Module, X_test:torch.Tensor, batch_size:int=8192):
    model.eval()
    final_probs = []
    for i in range(0,X_test.size(0),batch_size):
        logits = model(X_test[i : i+batch_size])
        batch_probs = torch.softmax(logits,dim=1)
        final_probs.append(batch_probs)

    return torch.cat(final_probs,dim=0)

@torch.inference_mode()
def validation_acc(model:nn.Module, X: torch.Tensor, y:torch.Tensor, batch_size:int = 8192):
    model.eval()
    correct, total = 0,0
    for i in range(0, X.size(0), batch_size):
        preds = model(X[i : i + batch_size]).argmax(1)
        correct += (preds == y[i:i+batch_size]).sum().item()
        total += preds.size(0)
    return correct/total

@torch.inference_mode()
def validation_loss(model:nn.Module, X_val:torch.Tensor, y_val:torch.Tensor, criterion:nn.Module, device, batch_size=8192):
    model.eval()
    batch_total = 0
    epoch_loss = torch.zeros((), device=device)
    for i in range(0,X_val.size(0), batch_size):
        X_batch = X_val[i:i+batch_size]
        y_batch = y_val[i:i+batch_size]
        output = model(X_batch)
        loss = criterion(output, y_batch)
        batch_seen = y_batch.size(0)
        epoch_loss = loss.detach() * batch_seen
        batch_total += batch_seen

    return (epoch_loss/batch_total).item()

@torch.no_grad()
def get_layernorm_output(model_raw, X, batch_size=8192):
    outputs = []
    for i in range(0, X.size(0), batch_size):
        batch_out = model_raw.layernorm(model_raw.layer1(X[i:i+batch_size]))
        outputs.append(batch_out)
    return torch.cat(outputs, dim=0)

@torch.inference_mode()
def test_models(model:nn.Module, X_test:torch.Tensor, y_test:torch.Tensor):
    model.eval()
    
    probs = chunked_probs(model, X_test)
    probs_cpu = probs.cpu().numpy()
    predictions = probs.argmax(1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    test_acc = float((predictions==y_true).mean())
    test_f1  = float(f1_score(y_true, predictions, average="macro"))
    try:
        if probs_cpu.shape[1] == 2:
            test_auc = float(roc_auc_score(y_true, probs_cpu[:, 1]))
        else:
            test_auc = float(roc_auc_score(y_true, probs_cpu, multi_class="ovr", average="macro"))
    except Exception:
        test_auc = 0.0

    results = {
            "model":        "PerTreeObliviousNAT",
            "test_acc":     test_acc,
            "test_auc":     test_auc,
            "test_f1":      test_f1,
        }
    return results