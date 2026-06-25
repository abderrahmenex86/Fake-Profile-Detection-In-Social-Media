import torch

from src.dataset import FeatureExtractor


def build_extractor(args):
    return FeatureExtractor(feature_type=args.feature, use_pca=args.pca, pca_components=args.pca_components)


def build_model(model_name, **kwargs):
    seed = 1337
    use_gpu = torch.cuda.is_available()
    use_cuml = kwargs.pop("use_cuml", False)

    if use_cuml and use_gpu:
        if model_name == "nb":
            print("[WARNING] cuML does not support GaussianNB. Falling back to Scikit-Learn CPU.")
            from sklearn.naive_bayes import GaussianNB

            return GaussianNB(var_smoothing=kwargs.get("var_smoothing", 1e-9))

        elif model_name == "lr":
            from cuml.linear_model import LogisticRegression as cuLR

            return cuLR(C=kwargs.get("C", 1.0), penalty=kwargs.get("penalty", "l2"), max_iter=2000)

        elif model_name == "svm":
            from cuml.svm import SVC as cuSVC

            return cuSVC(
                C=kwargs.get("C", 1.0),
                gamma=kwargs.get("gamma", "scale"),
                kernel=kwargs.get("kernel", "rbf"),
                probability=True,
            )

        elif model_name == "rf":
            from cuml.ensemble import RandomForestClassifier as cuRF

            return cuRF(
                n_estimators=kwargs.get("n_estimators", 100), max_depth=kwargs.get("max_depth", 10), random_state=seed
            )

    if model_name == "nb":
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB(var_smoothing=kwargs.get("var_smoothing", 1e-9))

    elif model_name == "lr":
        from sklearn.linear_model import LogisticRegression

        c_val = kwargs.get("C", 1.0)
        penalty_val = kwargs.get("penalty", "l2")

        if penalty_val == "none":
            c_val = float("inf")

        return LogisticRegression(
            C=c_val,
            max_iter=2000,
            random_state=seed,
        )

    elif model_name == "svm":
        from sklearn.svm import SVC

        return SVC(
            C=kwargs.get("C", 1.0),
            gamma=kwargs.get("gamma", "scale"),
            kernel=kwargs.get("kernel", "rbf"),
            probability=True,
            random_state=seed,
        )

    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 10),
            min_samples_split=kwargs.get("min_samples_split", 2),
            n_jobs=-1,
            random_state=seed,
        )

    elif model_name == "xgb":
        from xgboost import XGBClassifier

        xgb_device = "cuda" if use_gpu else "cpu"
        return XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            n_jobs=-1 if not use_gpu else None,
            tree_method="hist",
            device=xgb_device,
            random_state=seed,
            eval_metric="logloss",
        )

    raise ValueError(f"Model {model_name} not supported.")
