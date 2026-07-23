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



def verify_checkpoint(checkpoint:dict, input_dim:int, n_trees:int, depth:int, n_classes:int, batch_size:int, epochs:int):
    saved_config = checkpoint.get("config",{})
    current_config = {
        "input_dim":input_dim,
        "n_trees":n_trees,
        "depth":depth,
        "n_classes":n_classes,
        "batch_size": batch_size,
        "epochs":epochs
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

def atomic_save_json(history, path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w",encoding="utf-8") as f:
        json.dump(history,f,indent=2)
    os.replace(temporary, path)