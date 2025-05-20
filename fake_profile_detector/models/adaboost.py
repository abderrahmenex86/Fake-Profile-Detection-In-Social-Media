from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from experiments.config import RANDOM_SEED


def create_model(params):
    base = DecisionTreeClassifier(
        max_depth=int(params["base_max_depth"]), random_state=1337
    )
    model = AdaBoostClassifier(
        estimator=base,
        n_estimators=int(params["n_estimators"]),
        learning_rate=params["learning_rate"],
        random_state=RANDOM_SEED,
    )
    return model


def default_params():
    return {"n_estimators": 50, "learning_rate": 1.0, "base_max_depth": 1}


def param_space():
    return [
        {
            "name": "n_estimators",
            "type": "categorical",
            "categories": list(range(25, 201, 25)),
        },
        {"name": "learning_rate", "type": "continuous", "bounds": [0.01, 2.0]},
        {
            "name": "base_max_depth",
            "type": "categorical",
            "categories": list(range(1, 6)),
        },
    ]


def evaluate(model, X_val, y_val):
    return accuracy_score(y_val, model.predict(X_val))
