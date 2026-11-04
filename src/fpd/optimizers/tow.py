import numpy as np

from .base import Budget, Result


def optimize(objective, space, budget, seed):
    rng = np.random.default_rng(seed)
    state = Budget(budget)
    population = min(6, budget)
    iterations = max(0, (budget - population) // population)
    pop = np.array([space.sample(rng) for _ in range(population)])
    for _ in range(iterations):
        scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
        weights = scores / (scores.sum() + 1e-12)
        updated = pop.copy()
        for index in range(population):
            force = np.zeros(pop.shape[1])
            for other in range(population):
                if index != other:
                    force += rng.random(pop.shape[1]) * weights[other] * (pop[other] - pop[index])
            updated[index] = np.clip(pop[index] + force, 0, 1)
        pop = updated
    while state.count < budget:
        state.evaluate(objective, space.decode(space.sample(rng)))
    return Result(state.best_params, state.best_score, state.history)
