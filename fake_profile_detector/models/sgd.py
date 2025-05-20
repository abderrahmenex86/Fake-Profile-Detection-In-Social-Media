from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.config import RANDOM_SEED


def create_model(params):
    alpha = 1.0 / params["C"]

    model = SGDClassifier(
        alpha=alpha,
        loss="hinge",
        penalty="l2",
        learning_rate="optimal",
        max_iter=100,
        tol=1e-3,
        random_state=RANDOM_SEED,
        early_stopping=True,
    )

    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("estimator", model)])


def default_params():
    return {"C": 1.0, "gamma": 1e-1, "kernel": "linear"}


def param_space():
    return [
        {"name": "C", "type": "continuous", "bounds": [1e-1, 10]},
        {
            "name": "gamma",
            "type": "continuous",
            "bounds": [1e-5, 1],
        },
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)
