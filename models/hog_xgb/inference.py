#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent


def load_artifacts(root=ROOT):
    config = json.loads((root / "feature_config.json").read_text())
    labels = json.loads((root / "labels.json").read_text())
    scaler = joblib.load(root / "scaler.joblib")
    model = XGBClassifier()
    model.load_model(root / "model.ubj")
    return config, labels, scaler, model


def extract(path, config):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    height = config["image"]["resize_height"]
    width = config["image"]["resize_width"]
    small = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=config["hog"]["orientations"],
        pixels_per_cell=tuple(config["hog"]["pixels_per_cell"]),
        cells_per_block=tuple(config["hog"]["cells_per_block"]),
        block_norm=config["hog"]["block_norm"],
        feature_vector=config["hog"]["feature_vector"],
    ).astype(np.float32)


def predict(paths, config, labels, scaler, model):
    features = np.vstack([extract(path, config) for path in paths])
    scaled = scaler.transform(features).astype(np.float32)
    probabilities = model.predict_proba(scaled)
    results = []
    for row, path in enumerate(paths):
        index = int(probabilities[row].argmax())
        results.append(
            {
                "path": str(path),
                "label": labels["classes"][index],
                "confidence": float(probabilities[row, index]),
                "probabilities": {
                    name: float(probabilities[row, column])
                    for column, name in enumerate(labels["classes"])
                },
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config, labels, scaler, model = load_artifacts()
    results = predict(args.images, config, labels, scaler, model)
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        probabilities = "  ".join(
            f"{name}={value:.3f}" for name, value in result["probabilities"].items()
        )
        print(f"{result['path']}: {result['label']} ({result['confidence']:.3f})  {probabilities}")


if __name__ == "__main__":
    main()
