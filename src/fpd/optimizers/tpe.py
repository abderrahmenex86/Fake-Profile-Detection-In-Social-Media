import numpy as np
import optuna

from .base import Result


def optimize(objective, space, budget, seed):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective(space.suggest(trial)), n_trials=budget)
    history = []
    best = -np.inf
    for trial in study.trials:
        if trial.value is not None:
            best = max(best, trial.value)
            history.append(float(best))
    return Result(study.best_params, float(study.best_value), history)
