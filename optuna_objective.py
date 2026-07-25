import optuna
import torch
import torch.nn as nn
from config import TrainingConfig, trial_config
from model_layers import ObliviousNATNet
from optuna_training import optuna_training
from optuna_util import GlobalBest
from test_model import test_models


def objective(model:nn.Module, X_train:torch.Tensor, y_train, X_val, y_val, X_test, y_test, device, base_config:TrainingConfig, trial:optuna.Trial, global_best:GlobalBest):
    config = trial_config(trial, base_config)

    try: 
        t_bestval, t_beststate, t_bestepoch = optuna_training(model, X_train, y_train, X_val, y_val, device, config, trial)

        update_best_results = global_best.update(
            val_acc= t_bestval,
            state= t_beststate,
            config= config,
            trail_number= trial.number,
            best_epoch=t_bestepoch
        )
        trails_results = test_models(model, X_test, y_test)
        print()
        print(f"Obtained results for Trial {trial.number} \n")
        print(f"Test Acc: {trails_results["test_acc"]} \n Test Auc: {trails_results['test_auc']} \n F1 score: {trails_results['test_f1']}")

        if update_best_results:
            print(f"New global best Obtained at Trail: {trial.number}, Best Val: {t_bestval}")


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

