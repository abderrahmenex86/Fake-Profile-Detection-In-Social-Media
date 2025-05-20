from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.config import RANDOM_SEED


def create_model(params):
    model = MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=2000,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )
    return Pipeline([('scaler', StandardScaler(with_mean=False)), ('estimator', model)])


def default_params():
    return {
        "hidden_layer_sizes": [100, 200, 100],
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    }


def param_space():
    return [
        {
            "name": "hidden_layer_sizes",
            "type": "categorical",
            "categories": list([x, x * 2, x] for x in range(20, 101, 20)),
        },
        {
            "name": "activation",
            "type": "categorical",
            "categories": ["relu", "tanh", "logistic"],
        },
        {"name": "alpha", "type": "continuous", "bounds": [1e-5, 1e-2]},
        {"name": "learning_rate_init", "type": "continuous", "bounds": [1e-4, 1e-1]},
    ]


def evaluate(model, X_val, y_val):
    return accuracy_score(y_val, model.predict(X_val))
