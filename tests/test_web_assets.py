import json
from pathlib import Path

from fpd.data import CLASSES
from fpd.features import FEATURE_DIMS

WEB = Path(__file__).resolve().parents[1] / "web"
ASSETS = WEB / "assets"


def test_model_json_matches_pipeline():
    model = json.loads((ASSETS / "model.json").read_text())
    assert model["classes"] == CLASSES
    assert model["feature"]["dimensions"] == FEATURE_DIMS["hog"]
    assert model["scaler"]["features"] == FEATURE_DIMS["hog"]
    assert len(model["offsets"]) == len(CLASSES)
    assert len(model["trees"]) == 200
    assert all(len(node) == 4 for tree in model["trees"] for node in tree)
    assert any(node[0] < 0 for tree in model["trees"] for node in tree)


def test_scaler_binary_layout():
    model = json.loads((ASSETS / "model.json").read_text())
    features = model["scaler"]["features"]
    payload = (ASSETS / model["scaler"]["file"]).read_bytes()
    assert len(payload) == features * 2 * 4


def test_samples_exist():
    for name in ("bot", "cyborg", "real", "verified"):
        path = WEB / "samples" / f"{name}.png"
        assert path.stat().st_size > 1000
