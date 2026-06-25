def get_param_space(model_name):
    if model_name == "nb":
        return [{"name": "var_smoothing", "type": "continuous", "bounds": [1e-9, 1e-2]}]
    elif model_name == "lr":
        return [
            {"name": "C", "type": "continuous", "bounds": [0.01, 10.0]},
            {"name": "penalty", "type": "categorical", "categories": ["l2", "none"]},
        ]
    elif model_name == "svm":
        return [
            {"name": "C", "type": "continuous", "bounds": [0.1, 10.0]},
            {"name": "gamma", "type": "continuous", "bounds": [1e-4, 1.0]},
            {"name": "kernel", "type": "categorical", "categories": ["rbf", "linear"]},
        ]
    elif model_name == "rf":
        return [
            {"name": "n_estimators", "type": "categorical", "categories": [50, 100, 200, 300]},
            {"name": "max_depth", "type": "categorical", "categories": [5, 10, 20, 50]},
            {"name": "min_samples_split", "type": "categorical", "categories": [2, 5, 10]},
        ]
    elif model_name == "xgb":
        return [
            {"name": "n_estimators", "type": "categorical", "categories": [50, 100, 200, 300]},
            {"name": "max_depth", "type": "categorical", "categories": [3, 6, 9, 12]},
            {"name": "learning_rate", "type": "continuous", "bounds": [0.01, 0.3]},
        ]
    raise ValueError(f"Unknown model: {model_name }")
