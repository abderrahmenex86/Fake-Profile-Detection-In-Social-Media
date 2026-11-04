import numpy as np

from .base import Budget, Result


def optimize(objective, space, budget, seed):
    rng = np.random.default_rng(seed)
    state = Budget(budget)
    population = min(6, budget)
    generations = max(0, (budget - population) // population)
    pop = np.array([space.sample(rng) for _ in range(population)])
    scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
    for _ in range(generations):
        children = []
        for _ in range(population):
            first, second = rng.choice(population, 2, replace=False)
            winner = pop[first] if scores[first] >= scores[second] else pop[second]
            loser = pop[second] if scores[first] >= scores[second] else pop[first]
            alpha = rng.random()
            child = alpha * winner + (1 - alpha) * loser
            mask = rng.random(child.shape) < 0.2
            child[mask] = rng.random(mask.sum())
            children.append(np.clip(child, 0, 1))
        pop = np.array(children)
        scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
    while state.count < budget:
        state.evaluate(objective, space.decode(space.sample(rng)))
    return Result(state.best_params, state.best_score, state.history)
