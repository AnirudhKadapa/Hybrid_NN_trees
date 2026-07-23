import json, os
import torch
from pathlib import Path
from data import load_data
from config import TrainingConfig, parse_config
from model_layers import ObliviousNATNet


def main():
    config = parse_config()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    input_dim = X_train.shape[1]

    model = ObliviousNATNet(input_dim, config.n_classes, config.depth, config.n_trees)

    




    