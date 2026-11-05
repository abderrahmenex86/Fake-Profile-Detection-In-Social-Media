import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fpd.data import CLASSES, REVISION, load_split
from fpd.features import extract_one, load_features
from fpd.tracking import metrics

FEATURE_CONFIG = {
    "feature": "hog",
    "image": {
        "color_mode": "RGB",
        "resize_height": 600,
        "resize_width": 450,
        "resize_interpolation": "INTER_AREA",
    },
    "hog": {
        "orientations": 9,
        "pixels_per_cell": [16, 16],
        "cells_per_block": [2, 2],
        "block_norm": "L2-Hys",
        "feature_vector": True,
    },
    "dimensions": 34992,
    "scaler": "StandardScaler",
    "pca_components": None,
    "classifier": "xgboost.XGBClassifier",
}


def best_row(path):
    frame = pd.read_csv(path)
    subset = frame[(frame["feature"] == "hog_nopca") & (frame["model"] == "xgb")]
    if subset.empty:
        raise ValueError(f"no hog_nopca/xgb row in {path}")
    return subset.sort_values("test_macro_f1", ascending=False).iloc[0]


def scaled(scaler, matrix, chunk=1024):
    output = np.empty_like(matrix)
    for start in range(0, len(matrix), chunk):
        stop = min(start + chunk, len(matrix))
        output[start:stop] = scaler.transform(matrix[start:stop])
    return output


def predict_ms(model, matrix, repeats=20):
    sample = matrix[:1]
    model.predict_proba(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(sample)
    return (time.perf_counter() - start) * 1000 / repeats


def hog_ms(paths, repeats=20):
    sample = paths[:repeats]
    extract_one(sample[0], "hog")
    start = time.perf_counter()
    for path in sample:
        extract_one(path, "hog")
    return (time.perf_counter() - start) * 1000 / len(sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="~/downloads/datasets/x_fake_profile_detection")
    parser.add_argument("--cache-dir", default="~/downloads/datasets/x_fake_profile_detection/features")
    parser.add_argument("--results", default="results/final.csv")
    parser.add_argument("--output", default="models/hog_xgb")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    output = Path(args.output)

    reference = best_row(args.results)
    params = json.loads(reference["best_params"])
    seed = int(reference["seed"])
    print(f"[export] params={params} seed={seed}", flush=True)

    train_paths, train_y = load_split(data_dir, "train")
    val_paths, val_y = load_split(data_dir, "validation")
    train_x = load_features(train_paths, "hog", "train", cache_dir)
    val_x = load_features(val_paths, "hog", "validation", cache_dir)

    full_x = np.vstack([train_x, val_x])
    full_y = np.concatenate([train_y, val_y])
    del train_x, val_x

    scaler = StandardScaler().fit(full_x)
    full_scaled = scaled(scaler, full_x)
    del full_x
    print(f"[export] train matrix={full_scaled.shape}", flush=True)

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        tree_method="hist",
        max_bin=64,
        device=args.device,
        n_jobs=-1,
        random_state=seed,
        eval_metric="mlogloss",
        verbosity=0,
    )
    start = time.perf_counter()
    model.fit(full_scaled, full_y)
    train_time = time.perf_counter() - start
    print(f"[export] fit {train_time:.1f}s", flush=True)
    del full_scaled

    test_paths, test_y = load_split(data_dir, "test")
    test_x = load_features(test_paths, "hog", "test", cache_dir)
    test_scaled = scaled(scaler, test_x)
    predictions = model.predict(test_scaled)
    values = metrics(test_y, predictions, CLASSES)
    print(f"[export] test accuracy={values['accuracy']:.4f} macro_f1={values['macro_f1']:.4f}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / "model.ubj")
    joblib.dump(scaler, output / "scaler.joblib")

    served = XGBClassifier()
    served.load_model(output / "model.ubj")
    predict_time = predict_ms(served, test_scaled)
    extract_time = hog_ms(test_paths)
    print(f"[export] hog={extract_time:.1f}ms predict={predict_time:.2f}ms", flush=True)

    labels = {
        "classes": CLASSES,
        "class_to_index": {name: index for index, name in enumerate(CLASSES)},
        "dataset": "drveronika/x_fake_profile_detection",
        "revision": REVISION,
    }
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    (output / "feature_config.json").write_text(json.dumps(FEATURE_CONFIG, indent=2) + "\n")

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training": {
            "split": "train + validation",
            "train_size": int(len(full_y)),
            "seed": seed,
            "params": params,
            "device": args.device,
            "train_time_s": round(train_time, 2),
            "feature_dimensions": FEATURE_CONFIG["dimensions"],
        },
        "test": {
            "split": "test",
            "size": int(len(test_y)),
            "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"],
            "macro_precision": values["macro_precision"],
            "macro_recall": values["macro_recall"],
            "per_class": values["classification_report"],
            "confusion_matrix": values["confusion_matrix"],
        },
        "cost": {
            "model_size_mb": round((output / "model.ubj").stat().st_size / 1024**2, 4),
            "scaler_size_mb": round((output / "scaler.joblib").stat().st_size / 1024**2, 4),
            "hog_ms_per_image": round(extract_time, 1),
            "predict_ms_batch1": round(predict_time, 2),
            "end_to_end_ms_per_image": round(extract_time + predict_time, 1),
            "hardware": f"{platform.processor() or 'cpu'} ({platform.machine()})",
        },
        "reference_benchmark": {
            "source": "results/final.csv",
            "test_accuracy": float(reference["test_accuracy"]),
            "test_macro_f1": float(reference["test_macro_f1"]),
            "model_size_mb": float(reference["model_size_mb"]),
            "inference_ms_batch1": float(reference["inference_ms_batch1"]),
            "train_time_s": float(reference["train_time_s"]),
            "hardware": "NVIDIA RTX PRO 4000 Blackwell",
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[export] wrote {output}", flush=True)
    print(classification_report(test_y, predictions, target_names=CLASSES, digits=4), flush=True)


if __name__ == "__main__":
    main()
