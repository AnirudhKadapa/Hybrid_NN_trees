import time, os
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.amp import GradScaler
from config import TrainingConfig
from training_utils import get_parameters, atomic_save, verify_checkpoint, atomic_save_json
from evals import validation_acc

def training(model_raw:nn.Module, X_train:torch.Tensor, X_val:torch.Tensor, y_train:torch.Tensor, y_val:torch.Tensor, device, config:TrainingConfig):
    model_raw = model_raw.to(device)
    model_params = get_parameters(model_raw)

    try:
        model = torch.compile(model_raw)
    except Exception:
        model = model_raw

    optimizer = AdamW(model_params, lr=config.lr, weight_decay= config.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=(device=="cuda"))
    loss_criterion = nn.CrossEntropyLoss()

    N = X_train.size(0)
    N_new = N - (N % config.batch_size)
    best_val_accuracy = float("-inf")
    best_state = None
    patience = 0
    start_epoch = 1
    history = []
    

    ckpt = config.ckpt    
    if ckpt.exists():
        checkpoint = torch.load(ckpt, map_location="cpu",weights_only=False)
        verify_checkpoint(checkpoint, X_train.shape[1], config.n_trees, config.depth, config.n_classes, config.batch_size, config.epochs)        

        last_completed_epoch = checkpoint["last_epoch"]
        model_raw.load_state_dict(checkpoint["last_model_state"])
        scheduler.load_state_dict(checkpoint["last_scheduler_state"])
        optimizer.load_state_dict(checkpoint["last_optimizer_state"])
        scaler.load_state_dict(checkpoint["last_scaler_state"])
        best_val_accuracy = checkpoint["best_val_acc"]
        best_state = checkpoint["best_model_state"]
        patience = checkpoint["last_patience"]
        history = checkpoint["last_history_state"]
        start_epoch = last_completed_epoch + 1

        print(f"Resuming from last Completed Epoch {last_completed_epoch}")

        if checkpoint["completed"]==True:
            return (history,best_val_accuracy, best_state)

        if start_epoch > config.epochs:
            print("Checkpoint already completed")
            print(f"Finished Training for set epochs {config.epochs}")
            return (history, best_val_accuracy, best_state)

    t0 = time.perf_counter()
    stop_reason = None

    for epoch in range(start_epoch,config.epochs+1):
        model.train()
        perm = torch.randperm(N,device=device)[:N_new]
        epoch_loss = 0.0
        batch_total = 0
        nan_flag = False


        for i in range(0, N_new, config.batch_size):
            idx = perm[i : i + config.batch_size]
            
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                output = model(X_train[idx])
                loss = loss_criterion(output, y_train[idx])
            
            if not torch.isfinite(loss):
                print(f"loss values are either Nan or not finite at epoch {epoch}")
                nan_flag = True
                break

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),1.0)

            if not torch.isfinite(grad_norm):
                print(f"Non finite gradient detected at epoch {epoch}")
                nan_flag = True
                break

            scaler.step(optimizer)
            scaler.update()

            batch_size_seen = idx.numel()
            epoch_loss += loss.item() * batch_size_seen
            batch_total += batch_size_seen
        
        if nan_flag:
            print(f"Stopped training due to numerical instability at epoch {epoch}")
            stop_reason = "numerical_instability"
            break

        scheduler.step()

        avg_loss = epoch_loss/batch_total
        val_acc = validation_acc(model,X_val,y_val)
        
        history.append(
            {
                "epoch":epoch,
                "loss":avg_loss,
                "val_acc":val_acc
            }
        )
        improved = val_acc > best_val_accuracy + 1e-4

        if improved:
            best_val_accuracy = val_acc
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model_raw.state_dict().items()}
            patience = 0
        else:
            patience += 1


        last_model_state = {name:state_value.detach().cpu().clone() for name, state_value in model_raw.state_dict().items()}
        checkpoint = {
            "last_epoch":epoch,
            "last_model_state":last_model_state,
            "best_model_state":best_state,
            "last_optimizer_state":optimizer.state_dict(),
            "last_scheduler_state":scheduler.state_dict(),
            "last_scaler_state":scaler.state_dict(),
            "last_patience":patience,
            "best_val_acc":best_val_accuracy,
            "last_history_state":history,

            "completed": False,
            "stop_reason": stop_reason,

            "config": {
                "input_dim": X_train.shape[1],
                "n_trees": config.n_trees,
                "depth": config.depth,
                "n_classes": config.n_classes,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "patience": config.patience,
            },
        }

        atomic_save(checkpoint, config.ckpt)
        atomic_save_json(history, config.full_results)

        if improved or (epoch%5==0):
            elapsed = time.perf_counter() - t0
            print(f"  ep{epoch:4d}/{config.epochs}  loss={avg_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_val_accuracy:.4f}"
                  f"  {'★' if improved else ''}")

        if patience >= config.patience:
            print(f"Early stop at epoch {epoch}. Training not improving, patience {config.patience} reached")
            stop_reason = "early_stopping"
            break

    if stop_reason is None:
        stop_reason="max_epochs"

    if ckpt.exists():
        final_checkpoint = torch.load(ckpt, map_location="cpu",weights_only=False)
        if stop_reason=="max_epochs" or stop_reason=="early_stopping":
            final_checkpoint["completed"] = True
        final_checkpoint["stop_reason"] = stop_reason
        atomic_save(final_checkpoint,config.ckpt)
    
    if device=="cuda":
        torch.cuda.synchronize()
    
    return history, best_val_accuracy, best_state







