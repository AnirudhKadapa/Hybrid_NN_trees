import optuna
import torch
from config import TrainingConfig, trial_config
from model_layers import ObliviousNATNet
from optuna_training import optuna_training
from optuna_util import GlobalBest


def objective(input_dim:int, X_train:torch.Tensor, y_train, X_val, y_val, device, base_config:TrainingConfig, trial:optuna.Trial, global_best:GlobalBest):
    config = trial_config(trial, base_config)

    model = ObliviousNATNet(input_dim=input_dim, output_dim=config.n_classes, depth=config.depth, n_trees=config.n_trees, dropout=config.dropout)

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    trial.set_user_attr("params", parameter_count)
    

    try: 
        t_bestval, t_beststate, t_bestepoch, layer1_entropy, layer2_entropy = optuna_training(model, X_train, y_train, X_val, y_val, device, config, trial)

        trial.set_user_attr("layer1_entropy", layer1_entropy)
        trial.set_user_attr("layer2_entropy", layer2_entropy)
        
        update_best_results = global_best.update(
            val_acc= t_bestval,
            state= t_beststate,
            config= config,
            trial_number= trial.number,
            best_epoch=t_bestepoch
        )

        if update_best_results:
            print()
            print(f"New global best Obtained at Trail: {trial.number}, Best Val: {t_bestval} \n")


        return float(t_bestval)
    
    except torch.OutOfMemoryError as error:
        raise optuna.TrialPruned(
            "Cuda Out of Memoery"
        ) from error

    finally:
        del model

        if "t_beststate" in locals():
            del t_beststate

        if device=="cuda":
            torch.cuda.empty_cache() 

