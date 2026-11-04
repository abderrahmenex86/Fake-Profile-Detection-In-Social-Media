#!/usr/bin/env bash
set -euo pipefail

python -c "import cuml" 2>/dev/null || {
  echo "cuML is missing. Use the RAPIDS image or the conda fallback in docs/gpu.md."
  exit 1
}

python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import xgboost" 2>/dev/null || pip install xgboost
python -c "import optuna" 2>/dev/null || pip install optuna
python -c "import skimage" 2>/dev/null || pip install scikit-image
python -c "import cv2" 2>/dev/null || pip install opencv-python-headless
python -c "import pynvml" 2>/dev/null || pip install nvidia-ml-py
python -c "import psutil" 2>/dev/null || pip install psutil

pip install -e . --no-deps

python - <<'PY'
import cupy as cp
from cuml.ensemble import RandomForestClassifier

features = cp.random.rand(200, 20)
labels = cp.random.randint(0, 4, 200)
model = RandomForestClassifier(n_estimators=10)
model.fit(features, labels)
print("cuML OK")
PY

python - <<'PY'
import numpy as np
from xgboost import XGBClassifier

features = np.random.rand(200, 20).astype(np.float32)
labels = np.random.randint(0, 4, 200)
model = XGBClassifier(
    n_estimators=5,
    max_depth=3,
    tree_method="hist",
    device="cuda",
    eval_metric="mlogloss",
)
model.fit(features, labels)
print("XGBoost CUDA OK")
PY

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
