import json, os
from pathlib import Path
import torch
import torch.nn as nn


def get_parameters(model:nn.Module):
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or name.startswith("layernorm"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
        
    return [
        {"params": decay},
        {"params":no_decay,"weight_decay":0.0}
    ]

def atomic_save(checkpoint:dict, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)

@torch.inference_mode()
def validation_acc(model:nn.Module, X: torch.Tensor, y:torch.Tensor, batch_size:int = 8192):
    model.eval()
    correct, total = 0,0
    for i in range(0, X.size(0), batch_size):
        preds = model(X[i : i + batch_size]).argmax(1)
        correct += (preds == y[i:i+batch_size]).sum().item()
        total += batch_size
    return correct/total

def verify_checkpoint(checkpoint:dict, input_dim:int, n_trees:int, depth:int, n_classes:int, batch_size:int):
    saved_config = checkpoint.get("config",{})
    current_config = {
        "input_dim":input_dim,
        "n_trees":n_trees,
        "depth":depth,
        "n_classes":n_classes,
        "batch_size": batch_size,
    }

    for name, value in current_config.items():
        saved_value = saved_config.get(name)

        if saved_value is None:
            raise ValueError(
                f"Current Config has no values for {name}"
            )

        if saved_value != value:
            raise ValueError(
                f"Checkpoint value for {name}:{saved_value} mismatch with current entered {name}:{value}"
            )
        