import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image
from xgboost import XGBClassifier

from fpd.data import CLASSES
from fpd.features import FEATURE_DIMS

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "hog_xgb"


def load_inference():
    spec = importlib.util.spec_from_file_location("hog_xgb_inference", MODEL_DIR / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_matches_pipeline():
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    config = json.loads((MODEL_DIR / "feature_config.json").read_text())
    assert labels["classes"] == CLASSES
    assert config["dimensions"] == FEATURE_DIMS["hog"]
    assert config["pca_components"] is None


def test_artifacts_load_and_predict(tmp_path):
    inference = load_inference()
    config, labels, scaler, model = inference.load_artifacts(MODEL_DIR)
    assert scaler.n_features_in_ == FEATURE_DIMS["hog"]

    image = np.random.default_rng(0).integers(0, 255, (800, 600, 3), dtype=np.uint8)
    path = tmp_path / "profile.png"
    Image.fromarray(image).save(path)

    features = inference.extract(path, config)
    assert features.shape == (FEATURE_DIMS["hog"],)

    probabilities = model.predict_proba(scaler.transform(features.reshape(1, -1)))
    assert probabilities.shape == (1, len(CLASSES))
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


def test_model_is_a_booster():
    model = XGBClassifier()
    model.load_model(MODEL_DIR / "model.ubj")
    assert model.n_classes_ == len(CLASSES)
    assert (MODEL_DIR / "scaler.joblib").exists()
    joblib.load(MODEL_DIR / "scaler.joblib")
