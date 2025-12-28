import optuna
import math
from config import SEED
import torch
import random

from nn import do_nn_training


def optuna_objective(trial):
    # wie bei GA
    individual = [
        trial.suggest_int("num_layers", 1, 4),
        trial.suggest_categorical("base_units", [32, 64, 128]),
        trial.suggest_categorical("width_pattern", ["constant", "increasing", "decreasing"]),
        trial.suggest_categorical("activation", ["relu", "tanh", "elu"]),
        trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        trial.suggest_categorical("optimizer", ["adam", "rmsprop"]),
        trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        trial.suggest_float("dropout_rate", 0.0, 0.5),
        trial.suggest_float("l2_weight_decay", 0.0, 1e-2),
        trial.suggest_categorical("scaler", ["standard", "minmax", "none"]),
        trial.suggest_int("epochs", 10, 100),
    ]

    # wie bei GA
    torch.manual_seed(SEED)
    random.seed(SEED)

    acc = do_nn_training(individual)
    return acc