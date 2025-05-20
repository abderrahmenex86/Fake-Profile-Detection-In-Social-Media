from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from experiments.config import RANDOM_SEED


def create_model(params):
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def default_params():
    return {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }


def param_space():
    return [
        {
            "name": "n_estimators",
            "type": "categorical",
            "categories": list(range(50, 501, 50)),
        },
        {
            "name": "max_depth",
            "type": "categorical",
            "categories": list(range(5, 51, 5)),
        },
        {
            "name": "min_samples_split",
            "type": "categorical",
            "categories": list(range(2, 11)),
        },
        {
            "name": "min_samples_leaf",
            "type": "categorical",
            "categories": list(range(1, 11)),
        },
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)
