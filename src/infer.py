import json
import os

import joblib

from src.dataset import load_image_paths


def run_inference(run_dir):
    hp_path = os.path.join(run_dir, "hyperparameters.json")
    if not os.path.exists(hp_path):
        print(f"[ERROR] {hp_path } missing. Cannot infer automatically.")
        return

    with open(hp_path, "r") as f:
        hp = json.load(f)

    extractor_path = os.path.join(run_dir, "extractor.joblib")
    model_path = os.path.join(run_dir, "best_model.joblib")

    if not os.path.exists(extractor_path) or not os.path.exists(model_path):
        print("[ERROR] Missing joblib files. Train a model first.")
        return

    extractor = joblib.load(extractor_path)
    model = joblib.load(model_path)

    print(f"\n[INFO] Loaded Run: {run_dir }")
    print(f"[INFO] Settings: Model={hp ['model'].upper ()} | Feature={hp ['feature'].upper ()} | PCA={hp ['pca']}")
    print(f"[INFO] Interactive Mode | Type 'q' to quit.")
    print("=" * 60)

    label2name = {0: "BOT", 1: "CYBORG", 2: "REAL", 3: "VERIFIED"}

    while True:
        img_path = input("Path to profile image (.png) > ")
        if img_path.lower().strip() in ["q", "quit", "exit"]:
            break
        if not os.path.exists(img_path):
            print("[ERROR] File not found.")
            continue

        X = extractor.transform([img_path])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][pred] if hasattr(model, "predict_proba") else 1.0

        print(f"Prediction > {label2name .get (pred ,'Unknown')} (Confidence: {prob :.2f})\n")
