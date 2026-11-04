import gc
import json
import time
from pathlib import Path

import numpy as np

from .config import ensure_dirs, feature_label
from .data import CLASSES, limit_split, load_split, search_indices
from .features import fit_preprocessor, load_features, transform_preprocessor
from .models import build_model
from .optimizers import OPTIMIZERS
from .search_spaces import SPACES, Space
from .tracking import (
    append_row,
    gpu_used_mb,
    inference_ms,
    metrics,
    model_size_mb,
    param_count,
    ram_mb,
    reset_torch_peak,
    torch_peak_mb,
    write_metadata,
)


def _feature_config(feature):
    config = dict(feature)
    if config.get("pca") and "pca_components" not in config:
        config["pca_components"] = 1000
    return config


def _load(config, feature):
    train_paths, train_y = load_split(config["data_dir"], "train")
    val_paths, val_y = load_split(config["data_dir"], "validation")
    test_paths, test_y = load_split(config["data_dir"], "test")
    train_paths, train_y = limit_split(train_paths, train_y, config.get("limit_train"), config["seed"])
    val_paths, val_y = limit_split(val_paths, val_y, config.get("limit_val"), config["seed"])
    test_paths, test_y = limit_split(test_paths, test_y, config.get("limit_test"), config["seed"])
    name = feature["name"]
    train_x = load_features(train_paths, name, "train", config["cache_dir"])
    val_x = load_features(val_paths, name, "val", config["cache_dir"])
    test_x = load_features(test_paths, name, "test", config["cache_dir"])
    return train_x, train_y, val_x, val_y, test_x, test_y


def _search(config, feature, train_x, train_y, val_x):
    indices = search_indices(train_y, config["search_train_size"], config["seed"])
    train_subset = train_x[indices]
    train_labels = train_y[indices]
    scaler, pca, train_scaled = fit_preprocessor(
        train_subset,
        feature.get("pca_components"),
        config["seed"],
    )
    val_scaled = transform_preprocessor(val_x, scaler, pca)
    return train_scaled, train_labels, val_scaled


def _valid(model, feature):
    return not (model == "rbf_svm" and not feature.get("pca"))


def _gpu_mb(model_name):
    if model_name in {"mlp", "cnn"}:
        peak = torch_peak_mb()
        if peak is not None:
            return peak
    return gpu_used_mb()


def _run_one(config, label, model_name, optimizer_name, seed, search_x, search_y, search_val_x, val_y, eval_path, run_path):
    space = Space(SPACES[model_name])
    run_id = f"{label}__{model_name}__{optimizer_name}__{seed}"
    best = {"score": -np.inf, "model": None, "params": None}
    counter = {"index": 0}

    def objective(params):
        reset_torch_peak()
        model = build_model(
            model_name,
            params,
            seed,
            config["backend"],
            input_dim=search_x.shape[1],
            mlp_epochs=config["mlp_epochs"],
            mlp_batch_size=config["mlp_batch_size"],
        )
        start = time.perf_counter()
        model.fit(search_x, search_y)
        fit_time = time.perf_counter() - start
        prediction = model.predict(search_val_x)
        values = metrics(val_y, prediction, CLASSES)
        counter["index"] += 1
        append_row(
            eval_path,
            {
                "run_id": run_id,
                "feature": label,
                "model": model_name,
                "optimizer": optimizer_name,
                "seed": seed,
                "eval_index": counter["index"],
                "params": json.dumps(params),
                "val_macro_f1": values["macro_f1"],
                "val_accuracy": values["accuracy"],
                "fit_time_s": fit_time,
                "gpu_mem_mb": _gpu_mb(model_name),
                "ram_mb": ram_mb(),
            },
        )
        if values["macro_f1"] > best["score"]:
            best["score"] = values["macro_f1"]
            best["model"] = model
            best["params"] = dict(params)
        else:
            del model
        gc.collect()
        return values["macro_f1"]

    start = time.perf_counter()
    result = OPTIMIZERS[optimizer_name](objective, space, config["budget"], seed)
    total_time = time.perf_counter() - start
    history = result.history
    best_index = int(np.argmax(history)) + 1
    threshold = 0.95 * result.best_score
    evals_to_95 = next(
        (index + 1 for index, value in enumerate(history) if value >= threshold),
        len(history),
    )
    best_model = best["model"]
    if best_model is None:
        best_model = build_model(
            model_name,
            result.best_params,
            seed,
            config["backend"],
            input_dim=search_x.shape[1],
            mlp_epochs=config["mlp_epochs"],
            mlp_batch_size=config["mlp_batch_size"],
        )
        best_model.fit(search_x, search_y)
    best_values = metrics(val_y, best_model.predict(search_val_x), CLASSES)
    inference_1, inference_32 = inference_ms(best_model, search_val_x)
    append_row(
        run_path,
        {
            "run_id": run_id,
            "feature": label,
            "model": model_name,
            "optimizer": optimizer_name,
            "seed": seed,
            "best_val_macro_f1": result.best_score,
            "best_val_accuracy": best_values["accuracy"],
            "best_params": json.dumps(result.best_params),
            "total_time_s": total_time,
            "evals_to_best": best_index,
            "evals_to_95": evals_to_95,
            "model_size_mb": model_size_mb(best_model),
            "param_count": param_count(best_model),
            "inference_ms_batch1": inference_1,
            "inference_ms_batch32": inference_32,
            "gpu_mem_mb": _gpu_mb(model_name),
            "backend": config["backend"],
            "history": json.dumps(history),
        },
    )
    print(f"[run] {run_id} best={result.best_score:.4f} time={total_time:.1f}s", flush=True)
    del best_model
    best["model"] = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def features_only(config):
    ensure_dirs(config)
    for feature in config["features"]:
        feature = _feature_config(feature)
        label = feature_label(feature)
        train_x, _train_y, val_x, _val_y, test_x, _test_y = _load(config, feature)
        print(
            f"[features] {label} train={train_x.shape} val={val_x.shape} test={test_x.shape}",
            flush=True,
        )


def run(config):
    ensure_dirs(config)
    write_metadata(Path(config["results_dir"]) / "metadata.json", config)
    eval_path = Path(config["results_dir"]) / "evaluations.csv"
    run_path = Path(config["results_dir"]) / "runs.csv"

    for feature in config["features"]:
        feature = _feature_config(feature)
        label = feature_label(feature)
        train_x, train_y, val_x, val_y, test_x, _test_y = _load(config, feature)
        search_x, search_y, search_val_x = _search(config, feature, train_x, train_y, val_x)
        print(
            f"[feature] {label} train={search_x.shape} val={search_val_x.shape} test={test_x.shape}",
            flush=True,
        )

        for model_name in config["models"]:
            if not _valid(model_name, feature):
                continue
            for optimizer_name in config["optimizers"]:
                for seed in config["seeds"]:
                    _run_one(
                        config,
                        label,
                        model_name,
                        optimizer_name,
                        seed,
                        search_x,
                        search_y,
                        search_val_x,
                        val_y,
                        eval_path,
                        run_path,
                    )
