import numpy as np

from .base import Budget, Result


def optimize(objective, space, budget, seed):
    rng = np.random.default_rng(seed)
    state = Budget(budget)
    current = space.sample(rng)
    current_score = state.evaluate(objective, space.decode(current))
    temperature = 0.1
    for _ in range(budget - 1):
        candidate = np.clip(current + rng.normal(0, temperature, current.shape), 0, 1)
        score = state.evaluate(objective, space.decode(candidate))
        if score > current_score or rng.random() < np.exp((score - current_score) / max(temperature, 1e-9)):
            current = candidate
            current_score = score
        temperature *= 0.97
    return Result(state.best_params, state.best_score, state.history)
