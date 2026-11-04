import numpy as np

from .base import Budget, Result


def optimize(objective, space, budget, seed):
    rng = np.random.default_rng(seed)
    state = Budget(budget)
    population = min(6, budget)
    iterations = max(0, (budget - population) // population)
    pop = np.array([space.sample(rng) for _ in range(population)])
    velocity = rng.uniform(-0.1, 0.1, pop.shape)
    scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
    personal = pop.copy()
    personal_scores = scores.copy()
    best = pop[np.argmax(scores)].copy()
    best_score = float(scores.max())
    for _ in range(iterations):
        r1 = rng.random(pop.shape)
        r2 = rng.random(pop.shape)
        velocity = 0.7 * velocity + 1.5 * r1 * (personal - pop) + 1.5 * r2 * (best - pop)
        pop = np.clip(pop + velocity, 0, 1)
        scores = np.array([state.evaluate(objective, space.decode(vector)) for vector in pop])
        improved = scores > personal_scores
        personal[improved] = pop[improved]
        personal_scores[improved] = scores[improved]
        index = int(np.argmax(personal_scores))
        if personal_scores[index] > best_score:
            best = personal[index].copy()
            best_score = float(personal_scores[index])
    while state.count < budget:
        state.evaluate(objective, space.decode(space.sample(rng)))
    return Result(state.best_params, state.best_score, state.history)
