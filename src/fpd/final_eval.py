import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import _feature_config, _load
from .config import ensure_dirs, feature_label
from .data import CLASSES
from .features import fit_preprocessor, transform_preprocessor
from .models import build_model
from .tracking import (
    append_row,
    gpu_used_mb,
    inference_ms,
    metrics,
    model_size_mb,
    param_count,
    reset_torch_peak,
    torch_peak_mb,
)


def _gpu_mb(model_name):
    if model_name == "mlp":
        peak = torch_peak_mb()
        if peak is not None:
            return peak
    return gpu_used_mb()


def run(config):
    ensure_dirs(config)
    runs_path = Path(config["results_dir"]) / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(runs_path)
    runs = pd.read_csv(runs_path)
    output_path = Path(config["results_dir"]) / "final.csv"

    for feature in config["features"]:
        feature = _feature_config(feature)
        label = feature_label(feature)
        train_x, train_y, val_x, val_y, test_x, test_y = _load(config, feature)
        full_x = np.vstack([train_x, val_x])
        full_y = np.concatenate([train_y, val_y])
        scaler, pca, full_scaled = fit_preprocessor(
            full_x,
            feature.get("pca_components"),
            config["seed"],
        )
        test_scaled = transform_preprocessor(test_x, scaler, pca)

        for model_name in config["models"]:
            subset = runs[(runs["feature"] == label) & (runs["model"] == model_name)]
            if subset.empty:
                continue
            best = subset.sort_values("best_val_macro_f1", ascending=False).iloc[0]
            params = json.loads(best["best_params"])
            reset_torch_peak()
            model = build_model(
                model_name,
                params,
                best["seed"],
                config["backend"],
                input_dim=full_scaled.shape[1],
                mlp_epochs=config["mlp_epochs"],
                mlp_batch_size=config["mlp_batch_size"],
            )
            start = time.perf_counter()
            model.fit(full_scaled, full_y)
            train_time = time.perf_counter() - start
            values = metrics(test_y, model.predict(test_scaled), CLASSES)
            inference_1, inference_32 = inference_ms(model, test_scaled)
            append_row(
                output_path,
                {
                    "feature": label,
                    "model": model_name,
                    "optimizer": best["optimizer"],
                    "seed": best["seed"],
                    "best_params": json.dumps(params),
                    "train_size": int(len(full_y)),
                    "test_accuracy": values["accuracy"],
                    "test_macro_f1": values["macro_f1"],
                    "test_precision": values["macro_precision"],
                    "test_recall": values["macro_recall"],
                    "per_class": values["classification_report"],
                    "confusion_matrix": json.dumps(values["confusion_matrix"]),
                    "train_time_s": train_time,
                    "inference_ms_batch1": inference_1,
                    "inference_ms_batch32": inference_32,
                    "model_size_mb": model_size_mb(model),
                    "param_count": param_count(model),
                    "gpu_mem_mb": _gpu_mb(model_name),
                    "backend": config["backend"],
                },
            )
            print(
                f"[final] {label} {model_name} test_f1={values['macro_f1']:.4f}",
                flush=True,
            )
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
