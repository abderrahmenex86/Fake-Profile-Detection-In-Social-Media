import torch

from src.dataset import FeatureExtractor

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier as skRF
    from sklearn.linear_model import LogisticRegression as skLR
    from sklearn.svm import SVC as skSVC

    try:
        from cuml.svm import SVC as cuSVC

        if not hasattr(cuSVC, "__sklearn_tags__"):
            cuSVC.__sklearn_tags__ = lambda self: skSVC().__sklearn_tags__()
    except ImportError:
        pass

    try:
        from cuml.ensemble import RandomForestClassifier as cuRF

        if not hasattr(cuRF, "__sklearn_tags__"):
            cuRF.__sklearn_tags__ = lambda self: skRF().__sklearn_tags__()
    except ImportError:
        pass

    try:
        from cuml.linear_model import LogisticRegression as cuLR

        if not hasattr(cuLR, "__sklearn_tags__"):
            cuLR.__sklearn_tags__ = lambda self: skLR().__sklearn_tags__()
    except ImportError:
        pass
except Exception:
    pass


def build_extractor(args):
    return FeatureExtractor(feature_type=args.feature, use_pca=args.pca, pca_components=args.pca_components)


def build_model(model_name, **kwargs):
    seed = 1337

    force_cpu = kwargs.pop("force_cpu", False)
    use_gpu = torch.cuda.is_available() and not force_cpu
    use_cuml = kwargs.pop("use_cuml", False) and not force_cpu

    if use_cuml and use_gpu:
        if model_name == "nb":
            from sklearn.naive_bayes import GaussianNB

            return GaussianNB(var_smoothing=kwargs.get("var_smoothing", 1e-9))

        elif model_name == "lr":
            from cuml.linear_model import LogisticRegression as cuLR

            return cuLR(C=kwargs.get("C", 1.0), penalty=kwargs.get("penalty", "l2"), max_iter=2000)

        elif model_name == "svm":
            from cuml.svm import SVC as cuSVC
            from sklearn.calibration import CalibratedClassifierCV

            prob = kwargs.pop("probability", False)
            base_svm = cuSVC(
                C=kwargs.get("C", 1.0), gamma=kwargs.get("gamma", "scale"), kernel=kwargs.get("kernel", "rbf")
            )
            if prob:
                return CalibratedClassifierCV(estimator=base_svm, ensemble=False)
            return base_svm

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

        c_val, penalty_val = kwargs.get("C", 1.0), kwargs.get("penalty", "l2")
        if penalty_val == "none":
            c_val = float("inf")

        return LogisticRegression(C=c_val, max_iter=2000, random_state=seed)

    elif model_name == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        prob = kwargs.pop("probability", False)
        base_svm = SVC(
            C=kwargs.get("C", 1.0),
            gamma=kwargs.get("gamma", "scale"),
            kernel=kwargs.get("kernel", "rbf"),
            random_state=seed,
        )
        if prob:
            return CalibratedClassifierCV(estimator=base_svm, ensemble=False)
        return base_svm

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
