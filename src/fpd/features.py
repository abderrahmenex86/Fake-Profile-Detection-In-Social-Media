import hashlib
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

FEATURE_DIMS = {"hog": 34992, "sift": 32768, "raw": 90000}


def _read(path):
    image = imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def _hog(image):
    small = cv2.resize(image, (450, 600), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    ).astype(np.float32)


def _sift(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    target = 256 * 128
    output = np.zeros(target, dtype=np.float32)
    if descriptors is None:
        return output
    order = np.argsort([keypoint.response for keypoint in keypoints])[::-1]
    flat = descriptors[order][:256].flatten()
    output[: min(len(flat), target)] = flat[:target]
    return output


def _raw(image):
    small = resize(image, (150, 200, 3), anti_aliasing=True)
    return small.astype(np.float32).flatten()


def extract_one(path, name):
    if name == "hog":
        return _hog(_read(path))
    if name == "sift":
        return _sift(path)
    if name == "raw":
        return _raw(_read(path))
    raise ValueError(name)


def cache_path(cache_dir, name, split, paths):
    digest = hashlib.md5("\n".join(paths).encode("utf-8")).hexdigest()[:10]
    return Path(cache_dir) / f"{name}_{split}_{len(paths)}_{digest}.npy"


def load_features(paths, name, split, cache_dir, overwrite=False):
    path = cache_path(cache_dir, name, split, paths)
    if path.exists() and not overwrite:
        return np.load(path)
    rows = [extract_one(item, name) for item in tqdm(paths, desc=f"{name} {split}", disable=None)]
    matrix = np.vstack(rows).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def fit_preprocessor(train, pca_components=None, seed=1337):
    scaler = StandardScaler()
    transformed = scaler.fit_transform(train).astype(np.float32)
    pca = None
    if pca_components:
        components = min(pca_components, transformed.shape[0], transformed.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
        transformed = pca.fit_transform(transformed).astype(np.float32)
    return scaler, pca, transformed


def transform_preprocessor(matrix, scaler, pca):
    transformed = scaler.transform(matrix).astype(np.float32)
    if pca is not None:
        transformed = pca.transform(transformed).astype(np.float32)
    return transformed
