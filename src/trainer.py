import json
import os

import joblib
from tqdm import tqdm

from src.factory import build_model
from src.helpers import compute_metrics


def run_training(args, X_train, y_train, X_val, y_val, extractor, run_dir, best_params=None):
    params = best_params if best_params else {}
    model = build_model(args.model, **params)

    tqdm.write(f"\n[INFO] Training {args .model .upper ()}...")
    model.fit(X_train, y_train)

    tqdm.write("[INFO] Evaluating on Validation Set...")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_val, y_pred, y_prob)

    tqdm.write("[INFO] Saving Artifacts...")
    joblib.dump(extractor, os.path.join(run_dir, "extractor.joblib"))
    joblib.dump(model, os.path.join(run_dir, "best_model.joblib"))

    history = {"val": metrics}
    with open(os.path.join(run_dir, f"{args .model }_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    with open(os.path.join(run_dir, "architecture.txt"), "w") as f:
        f.write(repr(model))

    tqdm.write(f"[SUCCESS] Validation Accuracy: {metrics ['accuracy']:.4f}")
    tqdm.write(f"[SUCCESS] All artifacts saved to {run_dir }")
