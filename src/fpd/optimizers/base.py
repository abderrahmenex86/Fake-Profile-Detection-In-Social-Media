from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    best_params: dict
    best_score: float
    history: list


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []

    def evaluate(self, objective, params):
        if self.count >= self.limit:
            raise RuntimeError("budget exceeded")
        score = float(objective(params))
        self.count += 1
        if score > self.best_score:
            self.best_score = score
            self.best_params = dict(params)
        self.history.append(float(self.best_score))
        return score
