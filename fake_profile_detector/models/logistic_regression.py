from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.config import RANDOM_SEED


def create_model(params):
    model = LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver="saga",
        max_iter=10000,
        random_state=RANDOM_SEED,
        n_jobs=4,
    )
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("estimator", model)])


def default_params():
    return {"C": 1.0, "penalty": "l2"}


def param_space():
    return [
        {"name": "C", "type": "continuous", "bounds": [0.01, 10]},
        {"name": "penalty", "type": "categorical", "categories": ["l1", "l2"]},
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)
