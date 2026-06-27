import gc
import json
import os

import joblib
from tqdm import tqdm

from src.factory import build_model
from src.helpers import compute_metrics


def run_training(args, X_train, y_train, X_val, y_val, extractor, run_dir, best_params=None):
    params = best_params.copy() if best_params else {}

    if getattr(args, "use_cuml", False):
        params["use_cuml"] = True

    if args.model == "svm":
        params["probability"] = True

    if args.model == "xgb" and args.feature == "raw" and not getattr(args, "pca", False):
        params["force_cpu"] = True

    model = build_model(args.model, **params)
    tqdm.write(f"\n[INFO] Training {args.model.upper()}...")

    try:
        with joblib.parallel_backend("sequential"):
            model.fit(X_train, y_train)
    except Exception as e:
        err_msg = str(e).lower()
        is_cuda_err = any(x in err_msg for x in ["memory", "alloc", "cublas", "cuda"])

        if is_cuda_err:
            tqdm.write(f"\n[WARNING] GPU Memory Error encountered: {e}")
            tqdm.write("[INFO] Wiping VRAM and dynamically falling back to CPU (32GB RAM / 36 Cores)...")

            del model
            gc.collect()
            if getattr(args, "use_cuml", False):
                try:
                    import cupy as cp

                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except ImportError:
                    pass

            cpu_params = params.copy()
            cpu_params["force_cpu"] = True

            model = build_model(args.model, **cpu_params)

            with joblib.parallel_backend("sequential"):
                model.fit(X_train, y_train)
            tqdm.write("[SUCCESS] CPU fallback training completed successfully!")
        else:
            raise e

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
