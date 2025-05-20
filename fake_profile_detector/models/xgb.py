from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from experiments.config import RANDOM_SEED


def create_model(params):
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )


def default_params():
    return {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }


def param_space():
    return [
        {
            "name": "n_estimators",
            "type": "categorical",
            "categories": list(range(50, 301, 25)),
        },
        {"name": "max_depth", "type": "categorical", "categories": list(range(3, 13))},
        {"name": "learning_rate", "type": "continuous", "bounds": [0.01, 0.3]},
        {"name": "subsample", "type": "continuous", "bounds": [0.5, 1.0]},
        {"name": "colsample_bytree", "type": "continuous", "bounds": [0.5, 1.0]},
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)
