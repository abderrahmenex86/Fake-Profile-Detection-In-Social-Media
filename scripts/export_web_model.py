import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost
from xgboost import XGBClassifier

from fpd.data import CLASSES, load_split
from fpd.features import load_features


def parse_tree(text):
    root = json.loads(text)
    stack = [root]
    nodes = {}
    while stack:
        node = stack.pop()
        node_id = node["nodeid"]
        if "leaf" in node:
            nodes[node_id] = (-1, float(node["leaf"]), -1, -1)
            continue
        feature = int(node["split"][1:])
        nodes[node_id] = (feature, float(node["split_condition"]), int(node["yes"]), int(node["no"]))
        stack.extend(node["children"])
    return [list(nodes[index]) for index in range(len(nodes))]


def evaluate(trees, offsets, matrix):
    margins = np.zeros((len(matrix), len(offsets)), dtype=np.float64)
    for tree_index, nodes in enumerate(trees):
        class_index = tree_index % len(offsets)
        for row in range(len(matrix)):
            node = 0
            while True:
                feature, threshold, yes, no = nodes[node]
                if feature < 0:
                    margins[row, class_index] += threshold
                    break
                node = yes if matrix[row, feature] < threshold else no
    return margins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/hog_xgb")
    parser.add_argument("--output", default="web/assets")
    parser.add_argument("--data-dir", default="~/downloads/datasets/x_fake_profile_detection")
    parser.add_argument("--cache-dir", default="~/downloads/datasets/x_fake_profile_detection/features")
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = XGBClassifier()
    model.load_model(model_dir / "model.ubj")
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    classes = int(config["learner"]["learner_model_param"]["num_class"])
    trees = [parse_tree(text) for text in booster.get_dump(dump_format="json")]
    print(f"[web] trees={len(trees)} classes={classes}")

    scaler = joblib.load(model_dir / "scaler.joblib")
    paths, labels = load_split(Path(args.data_dir).expanduser(), "validation")
    matrix = load_features(paths[: args.rows], "hog", "validation", Path(args.cache_dir).expanduser())
    matrix = matrix[: args.rows]
    scaled = scaler.transform(matrix).astype(np.float32)

    margins = evaluate(trees, [0.0] * classes, scaled)
    reference = booster.predict(xgboost.DMatrix(scaled), output_margin=True)
    reference = np.asarray(reference).reshape(len(scaled), classes)
    offsets = (reference - margins).mean(axis=0)
    residual = np.abs(reference - margins - offsets).max()
    print(f"[web] class offsets={offsets.round(6).tolist()} residual={residual:.6f}")
    if residual > 1e-4:
        raise RuntimeError("tree evaluation does not match xgboost margins")

    probabilities = np.exp(margins + offsets)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    gap = np.abs(probabilities - model.predict_proba(scaled)).max()
    print(f"[web] probability gap vs xgboost={gap:.8f}")
    if gap > 1e-5:
        raise RuntimeError("tree evaluation does not match xgboost probabilities")

    payload = {
        "classes": CLASSES,
        "trees": trees,
        "offsets": [float(value) for value in offsets],
        "scaler": {
            "file": "scaler.bin",
            "features": int(scaler.n_features_in_),
            "layout": "float32 mean[features] then float32 scale[features], little-endian",
        },
        "feature": json.loads((model_dir / "feature_config.json").read_text()),
        "metrics": json.loads((model_dir / "metrics.json").read_text())["test"],
        "size": {
            "booster_kb": round((model_dir / "model.ubj").stat().st_size / 1024, 1),
            "trees": len(trees),
        },
    }
    (output / "model.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    scaler_bytes = np.concatenate(
        [scaler.mean_.astype("<f4"), scaler.scale_.astype("<f4")]
    ).tobytes()
    (output / "scaler.bin").write_bytes(scaler_bytes)
    print(f"[web] model.json={(output / 'model.json').stat().st_size / 1024:.1f} KB ", end="")
    print(f"scaler.bin={(output / 'scaler.bin').stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
