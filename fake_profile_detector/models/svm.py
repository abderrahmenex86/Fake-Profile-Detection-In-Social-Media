from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..configs.general import RANDOM_SEED


def create_model(params):
    model = SVC(
        C=params["C"],
        gamma=params["gamma"],
        kernel=params["kernel"],
        probability=True,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("estimator", model)])


def default_params():
    return {"C": 1.0, "gamma": 1e-1, "kernel": "rbf"}


def param_space():
    return [
        {"name": "C", "type": "continuous", "bounds": [1e-1, 10]},
        {"name": "gamma", "type": "continuous", "bounds": [1e-5, 1]},
        {
            "name": "kernel",
            "type": "categorical",
            "categories": ["linear", "rbf", "poly"],
        },
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)
