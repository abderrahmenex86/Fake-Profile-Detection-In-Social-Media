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
        order = rng.permutation(population)
        updated = pop.copy()
        for index in range(0, population - 1, 2):
            first = order[index]
            second = order[index + 1]
            winner = first if scores[first] >= scores[second] else second
            loser = second if scores[first] >= scores[second] else first
            step = rng.random(pop.shape[1]) * 0.5
            updated[loser] = np.clip(pop[loser] + step * (pop[winner] - pop[loser]), 0, 1)
            updated[winner] = np.clip(pop[winner] + rng.normal(0, 0.02, pop.shape[1]), 0, 1)
        pop = updated
        scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
    while state.count < budget:
        state.evaluate(objective, space.decode(space.sample(rng)))
    return Result(state.best_params, state.best_score, state.history)
