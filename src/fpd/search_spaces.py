import numpy as np

SPACES = {
    "logreg": [
        {"name": "C", "type": "float", "low": 1e-3, "high": 1e2, "log": True},
    ],
    "linear_svm": [
        {"name": "C", "type": "float", "low": 1e-3, "high": 1e2, "log": True},
    ],
    "rbf_svm": [
        {"name": "C", "type": "float", "low": 1e-2, "high": 1e2, "log": True},
        {"name": "gamma", "type": "float", "low": 1e-5, "high": 1e1, "log": True},
    ],
    "rf": [
        {"name": "n_estimators", "type": "cat", "choices": [100, 200, 300]},
        {"name": "max_depth", "type": "cat", "choices": [5, 10, 20, None]},
        {"name": "max_features", "type": "cat", "choices": ["sqrt", "log2", 1.0]},
    ],
    "xgb": [
        {"name": "n_estimators", "type": "cat", "choices": [50, 100, 200]},
        {"name": "max_depth", "type": "cat", "choices": [3, 6, 9, 12]},
        {"name": "learning_rate", "type": "float", "low": 1e-2, "high": 3e-1, "log": True},
        {"name": "subsample", "type": "float", "low": 0.6, "high": 1.0, "log": False},
        {"name": "colsample_bytree", "type": "float", "low": 0.6, "high": 1.0, "log": False},
    ],
    "mlp": [
        {"name": "hidden", "type": "cat", "choices": [128, 256, 512]},
        {"name": "dropout", "type": "cat", "choices": [0.1, 0.3, 0.5]},
        {"name": "lr", "type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        {"name": "weight_decay", "type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    ],
}


class Space:
    def __init__(self, specs):
        self.specs = specs

    def sample(self, rng):
        values = []
        for spec in self.specs:
            if spec["type"] == "cat":
                values.append(float(rng.integers(0, len(spec["choices"]))))
            else:
                values.append(float(rng.random()))
        return np.asarray(values, dtype=np.float64)

    def decode(self, vector):
        params = {}
        for value, spec in zip(vector, self.specs):
            if spec["type"] == "cat":
                index = int(np.clip(round(value), 0, len(spec["choices"]) - 1))
                params[spec["name"]] = spec["choices"][index]
            else:
                low = spec["low"]
                high = spec["high"]
                if spec["log"]:
                    position = np.log10(low) + value * (np.log10(high) - np.log10(low))
                    params[spec["name"]] = float(10**position)
                else:
                    params[spec["name"]] = float(low + value * (high - low))
        return params

    def suggest(self, trial):
        params = {}
        for spec in self.specs:
            if spec["type"] == "cat":
                params[spec["name"]] = trial.suggest_categorical(spec["name"], spec["choices"])
            elif spec["log"]:
                params[spec["name"]] = trial.suggest_float(spec["name"], spec["low"], spec["high"], log=True)
            else:
                params[spec["name"]] = trial.suggest_float(spec["name"], spec["low"], spec["high"])
        return params
