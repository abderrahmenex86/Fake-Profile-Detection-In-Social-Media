from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model(params):
    model = GaussianNB(var_smoothing=params["var_smoothing"])
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("estimator", model)])


def default_params():
    return {"var_smoothing": 1e-9}


def param_space():
    return [
        {"name": "var_smoothing", "type": "continuous", "bounds": [1e-12, 1e-6]},
    ]


def evaluate(model, X_val, y_val):
    return accuracy_score(y_val, model.predict(X_val))
