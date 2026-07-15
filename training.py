import time
import torch
import torch.nn as nn
from torch.amp import GradScaler 
from config import TrainingConfig

def get_parameter_groups(model:nn.Module):
    decay = []
    no_decay = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        
        if name.endswith(".bias") or name.startswith("layernorm.") :
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
            {"params": decay},
            {"params":no_decay,"weight_decay":0.0}
        ]
    
def train(model_raw: nn.Module, X_train, X_val, y_train, y_val, device, config: TrainingConfig):
    model_raw = model_raw.to(device)
    model_parameters = get_parameter_groups(model_raw)
    try:
        model = torch.compile(model_raw, mode="default")
    except Exception:
        model = model_raw
    
    optimizer = torch.optim.AdamW(model_parameters, weight_decay=config.weight_decay, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=(device=="cuda")) 
    loss_criterion = nn.CrossEntropyLoss()

    N = X_train.size(0)
    N_new = N - (N % config.batch_size)
    best_val_acc = -1
    best_state = None
    t0 = time.per_counter()

    for epoch in range(1,config.epochs+1):
        model.train()
        permute = torch.randperm(N,device=device)[:N_new]
        epoch_loss = 0.0
        n_batches = 0
        nan_flag = False

        for i in range(0,N_new,config.batch_size):
            idx = permute[i : i + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16, enabled=(device == "cuda")):
                loss = loss_criterion(model(X_train[idx]),y_train[idx])
            
            if torch.isnan(loss):
                print(f"Nan Flag Triggered for loss at epoch {epoch}")
                nan_flag = True
                break
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update
            epoch_loss += loss.item()
            n_batches += 1
        
        if nan_flag:
            break
        scheduler.step()


