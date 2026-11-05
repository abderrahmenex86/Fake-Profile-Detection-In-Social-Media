import csv
import gc
import os
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import psutil

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def timer():
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def gpu_used_mb():
    if not GPU_AVAILABLE:
        return None
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2


def torch_peak_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        return None
    return None


def reset_torch_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def free_gpu_memory():
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def model_size_mb(model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ubj") as handle:
            path = handle.name
        if hasattr(model, "get_booster"):
            model.get_booster().save_model(path)
        elif hasattr(model, "state_dict"):
            import torch

            torch.save(model.state_dict(), path)
        elif hasattr(model, "model") and hasattr(model.model, "state_dict"):
            import torch

            torch.save(model.model.state_dict(), path)
        else:
            joblib.dump(model, path)
        size = os.path.getsize(path) / 1024**2
        os.unlink(path)
        return size
    except Exception:
        return None


def param_count(model):
    if hasattr(model, "parameters"):
        return int(sum(parameter.numel() for parameter in model.parameters()))
    if hasattr(model, "model") and hasattr(model.model, "parameters"):
        return int(sum(parameter.numel() for parameter in model.model.parameters()))
    return None


def metrics(y_true, y_pred, labels):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    target = list(range(len(labels)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=target,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=target,
            target_names=labels,
            zero_division=0,
            digits=4,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=target).tolist(),
    }


def write_metadata(path, config):
    import importlib
    import json
    import platform

    payload = {
        "config": config,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ["sklearn", "xgboost", "cv2", "skimage", "torch", "cuml"]:
        try:
            module = importlib.import_module(name)
            payload[name] = getattr(module, "__version__", None)
        except Exception:
            payload[name] = None
    try:
        import torch

        payload["cuda"] = bool(torch.cuda.is_available())
        payload["gpu"] = torch.cuda.get_device_name(0) if payload["cuda"] else None
    except Exception:
        payload["cuda"] = None
        payload["gpu"] = None
    Path(path).write_text(json.dumps(payload, indent=2))


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def inference_ms(model, X, repeats=10):
    import numpy as np

    batch = X[:1]
    model.predict(batch)
    single = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(batch)
        single.append((time.perf_counter() - start) * 1000)
    batch = X[: min(32, len(X))]
    model.predict(batch)
    many = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(batch)
        many.append((time.perf_counter() - start) * 1000)
    return float(np.median(single)), float(np.median(many))
