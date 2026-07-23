import torch
import torch.nn as nn

@torch.inference_mode()
def chunked_probs(model:nn.Module, X_test:torch.Tensor, y_test:torch.Tensor, batch_size:int=8192):
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
