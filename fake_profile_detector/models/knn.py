from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model(params):
    model = KNeighborsClassifier(
        n_neighbors=int(params["n_neighbors"]),
        p=int(params["p"]),
        weights=params["weights"],
        n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("estimator", model)])


def default_params():
    return {"n_neighbors": 5, "p": 2, "weights": "uniform"}


def param_space():
    return [
        {
            "name": "n_neighbors",
            "type": "categorical",
            "categories": list(range(3, 18, 2)),
        },
        {"name": "p", "type": "categorical", "categories": list(range(1, 6, 1))},
        {
            "name": "weights",
            "type": "categorical",
            "categories": ["uniform", "distance"],
        },
    ]


def evaluate(model, X_val, y_val):
    return accuracy_score(y_val, model.predict(X_val))
