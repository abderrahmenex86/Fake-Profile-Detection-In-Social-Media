import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from xgboost import XGBClassifier

from fpd.data import CLASSES, load_split
from fpd.features import extract_one
from fpd.tracking import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="~/downloads/datasets/x_fake_profile_detection")
    parser.add_argument("--model-dir", default="models/hog_xgb")
    parser.add_argument("--output", default="/tmp/opencode/web-validation")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths, labels = load_split(Path(args.data_dir).expanduser(), "test")
    generator = np.random.default_rng(args.seed)
    chosen = np.sort(generator.choice(len(paths), size=args.count, replace=False))

    scaler = joblib.load(Path(args.model_dir) / "scaler.joblib")
    model = XGBClassifier()
    model.load_model(Path(args.model_dir) / "model.ubj")

    entries = []
    feature_rows = []
    gray_rows = []
    width = height = None
    with (output / "images.bin").open("wb") as images:
        for order, index in enumerate(chosen):
            path = paths[index]
            image = np.asarray(Image.open(path).convert("RGB"))
            height, width = image.shape[:2]
            rgba = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
            offset = images.tell()
            images.write(rgba.tobytes())

            small = cv2.resize(image, (450, 600), interpolation=cv2.INTER_AREA)
            gray_rows.append(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).tobytes())

            features = extract_one(path, "hog")
            feature_rows.append(features)
            probabilities = model.predict_proba(scaler.transform(features.reshape(1, -1)))[0]
            entries.append(
                {
                    "name": f"{CLASSES[int(labels[index])]}/{Path(path).name}",
                    "label": int(labels[index]),
                    "offset": offset,
                    "length": int(rgba.nbytes),
                    "probabilities": [float(value) for value in probabilities],
                }
            )
            if (order + 1) % 20 == 0:
                print(f"[validate] encoded {order + 1}/{len(chosen)}", flush=True)

    (output / "features.bin").write_bytes(np.vstack(feature_rows).astype("<f4").tobytes())
    (output / "gray.bin").write_bytes(b"".join(gray_rows))
    (output / "manifest.json").write_text(
        json.dumps({"classes": CLASSES, "entries": entries, "width": width, "height": height}, indent=2)
        + "\n"
    )

    reference = np.array([entry["probabilities"] for entry in entries])
    predicted = reference.argmax(axis=1)
    values = metrics(np.array([entry["label"] for entry in entries]), predicted, CLASSES)
    print(f"[validate] reference accuracy on sample={values['accuracy']:.4f}", flush=True)
    print(f"[validate] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
