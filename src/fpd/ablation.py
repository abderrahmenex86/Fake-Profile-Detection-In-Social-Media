import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import ensure_dirs
from .data import CLASSES, limit_split, load_split, search_indices
from .features import _hog, _read, cache_path, fit_preprocessor, transform_preprocessor
from .models import build_model
from .tracking import (
    append_row,
    free_gpu_memory,
    gpu_used_mb,
    inference_ms,
    metrics,
    model_size_mb,
    param_count,
    write_metadata,
)


def _variant_image(path, variant):
    image = _read(path)
    if variant == "full":
        return image
    if variant == "header":
        return image[:440]
    if variant == "bottom_mask":
        output = image.copy()
        output[440:] = output[440:].mean(axis=(0, 1), keepdims=True)
        return output
    raise ValueError(variant)


def _brightness(path):
    image = _read(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.asarray(
        [
            image[:, :, 0].mean(),
            image[:, :, 1].mean(),
            image[:, :, 2].mean(),
            gray.mean(),
            (gray > 240).mean(),
            np.abs(np.diff(gray, axis=1)).mean(),
        ],
        dtype=np.float32,
    )


def _variant_features(path, variant):
    if variant == "brightness":
        return _brightness(path)
    return _hog(_variant_image(path, variant))


def _load_variant(paths, variant, split, cache_dir):
    path = cache_path(cache_dir, f"ablation_{variant}", split, paths)
    if path.exists():
        return np.load(path)
    rows = [
        _variant_features(item, variant)
        for item in tqdm(paths, desc=f"{variant} {split}", disable=None)
    ]
    matrix = np.vstack(rows).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def _evaluate(model, x, y):
    prediction = model.predict(x)
    return metrics(y, prediction, CLASSES)


def run(config):
    ensure_dirs(config)
    write_metadata(Path(config["results_dir"]) / "metadata.json", config)
    train_paths, train_y = load_split(config["data_dir"], "train")
    val_paths, val_y = load_split(config["data_dir"], "validation")
    test_paths, test_y = load_split(config["data_dir"], "test")
    train_paths, train_y = limit_split(train_paths, train_y, config.get("limit_train"), config["seed"])
    val_paths, val_y = limit_split(val_paths, val_y, config.get("limit_val"), config["seed"])
    test_paths, test_y = limit_split(test_paths, test_y, config.get("limit_test"), config["seed"])
    indices = search_indices(train_y, config.get("search_train_size", len(train_y)), config["seed"])
    output_path = Path(config["results_dir"]) / "ablation.csv"

    for variant in config.get("ablation_variants", ["full", "header", "bottom_mask", "brightness"]):
        train_x = _load_variant(train_paths, variant, "train", config["cache_dir"])
        val_x = _load_variant(val_paths, variant, "val", config["cache_dir"])
        test_x = _load_variant(test_paths, variant, "test", config["cache_dir"])
        scaler, pca, train_scaled = fit_preprocessor(
            train_x[indices],
            None,
            config["seed"],
        )
        val_scaled = transform_preprocessor(val_x, scaler, pca)
        test_scaled = transform_preprocessor(test_x, scaler, pca)
        models = ["logreg", "rf"] if variant == "brightness" else ["xgb"]
        for model_name in models:
            if model_name == "xgb":
                params = {
                    "n_estimators": 100,
                    "max_depth": 8,
                    "learning_rate": 0.12,
                    "subsample": 0.79,
                    "colsample_bytree": 0.98,
                }
            elif model_name == "logreg":
                params = {"C": 1.0}
            else:
                params = {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"}
            for seed in config["seeds"]:
                free_gpu_memory()
                model = build_model(
                    model_name,
                    params,
                    seed,
                    config["backend"],
                    input_dim=train_scaled.shape[1],
                    force_cpu=model_name == "xgb",
                )
                start = time.perf_counter()
                model.fit(train_scaled, train_y[indices])
                train_time = time.perf_counter() - start
                val_values = _evaluate(model, val_scaled, val_y)
                test_values = _evaluate(model, test_scaled, test_y)
                inference_1, inference_32 = inference_ms(model, test_scaled)
                append_row(
                    output_path,
                    {
                        "variant": variant,
                        "model": model_name,
                        "seed": seed,
                        "val_macro_f1": val_values["macro_f1"],
                        "test_accuracy": test_values["accuracy"],
                        "test_macro_f1": test_values["macro_f1"],
                        "per_class": test_values["classification_report"],
                        "confusion_matrix": json.dumps(test_values["confusion_matrix"]),
                        "train_time_s": train_time,
                        "inference_ms_batch1": inference_1,
                        "inference_ms_batch32": inference_32,
                        "model_size_mb": model_size_mb(model),
                        "param_count": param_count(model),
                        "gpu_mem_mb": gpu_used_mb(),
                        "backend": config["backend"],
                    },
                )
                print(
                    f"[ablation] {variant} {model_name} seed={seed} test_f1={test_values['macro_f1']:.4f}",
                    flush=True,
                )
                del model
                free_gpu_memory()
