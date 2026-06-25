# src/dataset.py
import os

import cv2
import numpy as np
from PIL import Image, ImageFile
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FeatureExtractor:
    def __init__(self, feature_type="raw", use_pca=False, pca_components=500, batch_size=500):
        self.feature_type = feature_type
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.batch_size = max(batch_size, pca_components) if use_pca else batch_size

        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=pca_components) if use_pca else None

        if feature_type == "sift":
            self.sift = cv2.SIFT_create()

    def __getstate__(self):
        """Called by joblib.dump(). Removes unpicklable C++ objects."""
        state = self.__dict__.copy()
        if "sift" in state:
            del state["sift"]
        return state

    def __setstate__(self, state):
        """Called by joblib.load(). Restores the C++ objects."""
        self.__dict__.update(state)
        if self.feature_type == "sift":
            self.sift = cv2.SIFT_create()

    def _extract_single(self, path):
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            tqdm.write(f"[WARNING] Corrupted/truncated image at {path}: {e}. Falling back to black placeholder.")
            img = np.zeros((150, 200, 3), dtype=np.uint8)  # Consistent fallback dimensions

        img_resized = cv2.resize(img, (200, 150))

        if self.feature_type == "raw":
            return img_resized.flatten().astype(np.float32)

        elif self.feature_type == "hog":
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            from skimage.feature import hog

            return hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)).astype(np.float32)

        elif self.feature_type == "sift":
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            _, des = self.sift.detectAndCompute(gray, None)
            target_features = 234 * 128
            features = np.zeros(target_features, dtype=np.float32)
            if des is not None:
                flat_des = des.flatten()
                limit = min(len(flat_des), target_features)
                features[:limit] = flat_des[:limit]
            return features

        raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def fit_transform(self, image_paths):
        tqdm.write(f"[INFO] Extracting {self.feature_type.upper()} features (PCA: {self.use_pca})...")

        if self.use_pca and len(image_paths) < self.pca_components:
            tqdm.write(
                f"[WARNING] Dataset size ({len(image_paths)}) < PCA components ({self.pca_components}). Capping components dynamically."
            )
            self.pca_components = len(image_paths)
            self.batch_size = len(image_paths)
            self.pca = IncrementalPCA(n_components=self.pca_components)

        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Fitting Pipeline", leave=False):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_features = np.array([self._extract_single(p) for p in batch_paths])

            self.scaler.partial_fit(batch_features)
            if self.use_pca:
                if len(batch_features) >= self.pca_components:
                    scaled_batch = self.scaler.transform(batch_features)
                    self.pca.partial_fit(scaled_batch)

        all_features = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Transforming Data", leave=False):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_features = np.array([self._extract_single(p) for p in batch_paths])

            scaled = self.scaler.transform(batch_features)
            if self.use_pca:
                scaled = self.pca.transform(scaled)
            all_features.append(scaled)

        return np.vstack(all_features)

    def transform(self, image_paths):
        all_features = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Transforming Data", leave=False):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_features = np.array([self._extract_single(p) for p in batch_paths])
            scaled = self.scaler.transform(batch_features)
            if self.use_pca:
                scaled = self.pca.transform(scaled)
            all_features.append(scaled)
        return np.vstack(all_features)


def load_image_paths(dataset_dir="dataset", split="train"):
    paths, labels = [], []
    split_dir = os.path.join(dataset_dir, split)
    label2idx = {"BOT": 0, "CYBORG": 1, "REAL": 2, "VERIFIED": 3}

    for label_str, idx in label2idx.items():
        class_dir = os.path.join(split_dir, label_str)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(".png"):
                paths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return paths, np.array(labels)
