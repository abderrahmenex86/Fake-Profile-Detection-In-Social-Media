import numpy as np

from .base import Budget, Result


def optimize(objective, space, budget, seed):
    rng = np.random.default_rng(seed)
    state = Budget(budget)
    population = min(6, budget)
    iterations = max(0, (budget - population) // population)
    pop = np.array([space.sample(rng) for _ in range(population)])
    scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
    for _ in range(iterations):
        best = pop[np.argmax(scores)].copy()
        updated = pop.copy()
        for index in range(population):
            dice = int(rng.integers(1, 7, size=3).sum())
            scale = dice / 18
            direction = best - pop[index]
            updated[index] = np.clip(pop[index] + scale * direction * rng.random(pop.shape[1]), 0, 1)
        pop = updated
        scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
    while state.count < budget:
        state.evaluate(objective, space.decode(space.sample(rng)))
    return Result(state.best_params, state.best_score, state.history)
