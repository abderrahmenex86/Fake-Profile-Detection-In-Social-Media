import numpy as np
from PIL import Image

from fpd.features import extract_one, fit_preprocessor, transform_preprocessor


def _write_image(path, color):
    Image.new("RGB", (600, 800), color).save(path)


def test_feature_shapes(tmp_path):
    path = tmp_path / "sample.png"
    _write_image(path, (128, 64, 32))
    assert extract_one(path, "hog").shape == (34992,)
    assert extract_one(path, "sift").shape == (32768,)
    assert extract_one(path, "raw").shape == (90000,)


def test_preprocessor_shapes():
    rng = np.random.default_rng(0)
    train = rng.random((20, 100), dtype=np.float32)
    scaler, pca, transformed = fit_preprocessor(train, 10, 0)
    assert transformed.shape == (20, 10)
    test = rng.random((5, 100), dtype=np.float32)
    assert transform_preprocessor(test, scaler, pca).shape == (5, 10)
