import time
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.amp import GradScaler
import optuna
from config import TrainingConfig
from training_utils import get_parameters
from evals import validation_acc


def optuna_training(model_raw:nn.Module, X_train:torch.Tensor, y_train:torch.Tensor, X_val:torch.Tensor, y_val:torch.Tensor, device, config:TrainingConfig, trial:optuna.Trial):
    torch._dynamo.reset()
    model_raw = model_raw.to(device)
    model_params = get_parameters(model_raw)

    try:
        model = torch.compile(model_raw, mode="reduce-overhead")
    except Exception:
        model = model_raw
    
    optimizer = AdamW(model_params, lr=config.lr, weight_decay=config.weight_decay, fused=(device=="cuda"))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=(device=="cuda"))
    loss_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    N = X_train.size(0)
    N_new = N - (N % config.batch_size)
    best_val_accuracy = float("-inf")
    patience = 0
    best_epoch = 0
    best_state = None

    print(f"Trial : {trial.number:03d}")
    print(f"Num_trees: {config.n_trees}, Depth: {config.depth}, learning rate: {config.lr}, WD: {config.weight_decay}, LS:{config.label_smoothing}")
    print()
    for epoch in range(1, config.epochs+1):
        model.train()
        perm = torch.randperm(N,device=device)[:N_new]
        epoch_loss = torch.zeros((), device=device)
        batch_total = 0
        nan_flag = False

        for i in range(0,N_new,config.batch_size):
            idx = perm[i: i+ config.batch_size]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device,dtype=torch.float16, enabled=(device=="cuda")):
                output = model(X_train[idx])
                loss = loss_criterion(output, y_train[idx])

            if torch.isnan(loss):
                print(f"loss values are either Nan or not finite at epoch {epoch}")
                nan_flag = True
                break
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size_seen = idx.numel()
            epoch_loss += loss.detach() * batch_size_seen
            batch_total += batch_size_seen

        if nan_flag:
            raise optuna.TrialPruned(
                f"Traning Stopped, Trail {trial.number} pruned due to Numberical instability at epoch {epoch}"
            )

        scheduler.step()

        avg_loss = (epoch_loss/batch_total).item()
        val_acc = validation_acc(model,X_val,y_val)

        improved = val_acc > best_val_accuracy + 1e-4
        
        if improved:
            best_val_accuracy = val_acc
            best_state = {name: tensor.clone() for name, tensor in model_raw.state_dict().items()}
            patience = 0
            best_epoch = epoch
        else:
            patience += 1

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss {avg_loss:.5f} | "
            f"Val {val_acc:.5f} | "
            f"Best {best_val_accuracy:.5f}"
            f"  {'★' if improved else ''}"
        )

        trial.report(
            val_acc,
            step=epoch,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

        if patience >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Training  finished without producing a valid model state."
        )

    return best_val_accuracy, best_state, best_epoch 


