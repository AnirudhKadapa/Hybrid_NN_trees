import time
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.amp import GradScaler
from config import TrainingConfig

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

@torch.inference_mode()
def validation_acc(model:nn.Module, X: torch.Tensor, y:torch.Tensor, batch_size:int = 8192):
    model.eval()
    correct, total = 0,0
    for i in range(0, X.size(0), batch_size):
        preds = model(X[i : i + batch_size]).argmax(1)
        correct += (preds == y[i:i+batch_size]).sum().item()
        total += batch_size
    return correct/total

def training(
        model_raw:nn.Module, 
        X_train:torch.Tensor, 
        X_val:torch.Tensor, 
        y_train:torch.Tensor, 
        y_val:torch.Tensor, 
        device, 
        config:TrainingConfig
        ):
    model_raw = model_raw.to(device)
    model_params = get_parameters(model)

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
    best_state = []
    patience = 0
    t0 = time.perf_counter()

    for epoch in range(1,config.epochs+1):
        model.train()
        perm = torch.randperm(N,device=device)[:N_new]
        epoch_loss = 0.0
        n_batches = 0
        nan_flag = False
        history = []

        for i in range(0, N_new, config.batch_size):
            idx = perm[i : i + config.batch_size]
            
            optimizer.zero_grad()

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
            break
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

        if improved or (epoch%10==0):
            elapsed = time.perf_counter() - t0
            print(f"  ep{epoch:4d}/{config.epochs}  loss={avg_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_val_accuracy:.4f}"
                  f"  {'★' if improved else ''}")

        if patience > config.patience:
            print(f"Early stop at epoch {epoch}. Training not improving, patience {config.patience} reached")
            break

    if device=="cuda":
        torch.cuda.synchronize()
    
    return history, best_val_accuracy, best_state







