import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from fpd.data import CLASSES, limit_split, load_split
from fpd.features import fit_preprocessor, transform_preprocessor
from fpd.models import build_model
from fpd.tracking import append_row, free_gpu_memory, metrics, model_size_mb, param_count


def sift_descriptors(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    return keypoints, descriptors


def first_descriptors(descriptors, count):
    target = count * 128
    output = np.zeros(target, dtype=np.float32)
    if descriptors is not None:
        flat = descriptors[:count].flatten()
        output[: min(len(flat), target)] = flat[:target]
    return output


def top_descriptors(keypoints, descriptors, count):
    target = count * 128
    output = np.zeros(target, dtype=np.float32)
    if descriptors is not None and keypoints:
        order = np.argsort([keypoint.response for keypoint in keypoints])[::-1]
        flat = descriptors[order][:count].flatten()
        output[: min(len(flat), target)] = flat[:target]
    return output


def bovw_descriptor(descriptors, kmeans, clusters):
    output = np.zeros(clusters, dtype=np.float32)
    if descriptors is not None and len(descriptors):
        labels = kmeans.predict(descriptors)
        counts = np.bincount(labels, minlength=clusters).astype(np.float32)
        total = counts.sum()
        if total > 0:
            output = counts / total
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="~/downloads/datasets/x_fake_profile_detection")
    parser.add_argument("--output", default="results/sift_investigation.csv")
    parser.add_argument("--limit-train", type=int, default=2000)
    parser.add_argument("--limit-val", type=int, default=1000)
    parser.add_argument("--limit-test", type=int, default=4000)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    data_dir = str(Path(args.data_dir).expanduser())
    train_paths, train_y = load_split(data_dir, "train")
    val_paths, val_y = load_split(data_dir, "validation")
    test_paths, test_y = load_split(data_dir, "test")
    train_paths, train_y = limit_split(train_paths, train_y, args.limit_train, args.seed)
    val_paths, val_y = limit_split(val_paths, val_y, args.limit_val, args.seed)
    test_paths, test_y = limit_split(test_paths, test_y, args.limit_test, args.seed)
    print(f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}", flush=True)

    sample_descriptors = []
    for path in tqdm(train_paths[: args.sample], desc="codebook"):
        _, descriptors = sift_descriptors(path)
        if descriptors is not None:
            sample_descriptors.append(descriptors)
    if not sample_descriptors:
        raise RuntimeError("no SIFT descriptors in codebook sample")
    sample = np.vstack(sample_descriptors)
    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=args.seed,
        batch_size=4096,
        n_init=3,
    )
    kmeans.fit(sample)
    print(f"codebook descriptors={len(sample)} clusters={args.clusters}", flush=True)

    splits = [("train", train_paths), ("val", val_paths), ("test", test_paths)]
    features = {"first234": [], "top256": [], "bovw256": []}
    for split, paths in splits:
        for path in tqdm(paths, desc=split):
            keypoints, descriptors = sift_descriptors(path)
            features["first234"].append(first_descriptors(descriptors, 234))
            features["top256"].append(top_descriptors(keypoints, descriptors, 256))
            features["bovw256"].append(bovw_descriptor(descriptors, kmeans, args.clusters))

    matrices = {name: np.vstack(values).astype(np.float32) for name, values in features.items()}
    train_size = len(train_y)
    val_size = len(val_y)
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    for name, matrix in matrices.items():
        train_x = matrix[:train_size]
        val_x = matrix[train_size : train_size + val_size]
        test_x = matrix[train_size + val_size :]
        scaler, pca, train_scaled = fit_preprocessor(train_x, None, args.seed)
        val_scaled = transform_preprocessor(val_x, scaler, pca)
        test_scaled = transform_preprocessor(test_x, scaler, pca)

        for model_name in ["logreg", "xgb"]:
            params = {"C": 1.0} if model_name == "logreg" else xgb_params
            free_gpu_memory()
            model = build_model(model_name, params, args.seed, "gpu")
            start = time.perf_counter()
            model.fit(train_scaled, train_y)
            train_time = time.perf_counter() - start
            val_values = metrics(val_y, model.predict(val_scaled), CLASSES)
            test_values = metrics(test_y, model.predict(test_scaled), CLASSES)
            append_row(
                args.output,
                {
                    "feature": name,
                    "model": model_name,
                    "val_accuracy": val_values["accuracy"],
                    "val_macro_f1": val_values["macro_f1"],
                    "test_accuracy": test_values["accuracy"],
                    "test_macro_f1": test_values["macro_f1"],
                    "train_time_s": train_time,
                    "model_size_mb": model_size_mb(model),
                    "param_count": param_count(model),
                },
            )
            print(
                f"[sift] {name} {model_name} val={val_values['macro_f1']:.4f} test={test_values['macro_f1']:.4f}",
                flush=True,
            )
            del model
            free_gpu_memory()


if __name__ == "__main__":
    main()
