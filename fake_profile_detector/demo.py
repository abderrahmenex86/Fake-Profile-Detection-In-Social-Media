import glob
import os
import pickle

from .configs.general import SAVE_DIR
from .utils.data_loader import load_data
from .utils.metrics import compute_metrics


def load_saved_models():
    models_dir = os.path.join(SAVE_DIR, "models")

    if not os.path.exists(models_dir):
        raise ValueError(f"Models directory not found: {models_dir}")

    models = {}
    model_files = glob.glob(os.path.join(models_dir, "*_tow.pkl"))

    if not model_files:
        raise ValueError(f"No TOW-optimized models found in {models_dir}")

    print(f"Loading models from {models_dir}...")

    for model_file in model_files:
        model_name = os.path.basename(model_file).replace("_tow.pkl", "")

        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            models[model_name] = model
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")

    return models


def main():
    (
        _,
        _,
        _,
        _,
        _,
        _,
        X,
        y,
    ) = load_data("x")

    models = load_saved_models()

    for model_name in models:
        print(f"Running {model_name}")
        model_module = models[model_name]

        y_pred = model_module.predict(X)
        y_prob = (
            model_module.predict_proba(X)
            if hasattr(model_module, "predict_proba")
            else None
        )

        print(f"Metrics for {model_name}:")
        metrics = compute_metrics(y, y_pred, y_prob)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
