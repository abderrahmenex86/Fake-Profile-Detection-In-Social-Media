import lightgbm as lgb
from sklearn.metrics import accuracy_score

from experiments.config import RANDOM_SEED


def create_model(params):
    model = lgb.LGBMClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=params["learning_rate"],
        num_leaves=int(params["num_leaves"]),
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    return model


def default_params():
    return {
        "n_estimators": 100,
        "max_depth": 7,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
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
            "categories": list(range(3, 16, 2)),
        },
        {"name": "learning_rate", "type": "continuous", "bounds": [0.01, 0.3]},
        {
            "name": "num_leaves",
            "type": "categorical",
            "categories": list(range(11, 102, 10)),
        },
        {"name": "subsample", "type": "continuous", "bounds": [0.5, 1.0]},
        {"name": "colsample_bytree", "type": "continuous", "bounds": [0.5, 1.0]},
    ]


def evaluate(model, X_val, y_val):
    return accuracy_score(y_val, model.predict(X_val))
