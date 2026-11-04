import shutil
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["BOT", "CYBORG", "REAL", "VERIFIED"]
REPO_ID = "drveronika/x_fake_profile_detection"
REVISION = "28bc1fbee3aa928ab3d29d9766155a5233465216"
SPLITS = ["train", "validation", "test"]
EXPECTED = {"train": 10000, "validation": 1000, "test": 4000}


def download(data_dir):
    root = Path(data_dir)
    for split in SPLITS:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        archive = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{split}.zip",
            repo_type="dataset",
            revision=REVISION,
        )
        temp = root / f"temp_{split}"
        if temp.exists():
            shutil.rmtree(temp)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temp)
        counters = {label: 0 for label in CLASSES}
        for source in temp.rglob("*"):
            if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            parts = [part.lower() for part in source.parts]
            label = next((name for name in CLASSES if name.lower() in parts), None)
            if label is None:
                continue
            target = split_dir / label
            target.mkdir(parents=True, exist_ok=True)
            destination = target / f"{counters[label]:05d}.png"
            counters[label] += 1
            Image.open(source).convert("RGB").save(destination)
        shutil.rmtree(temp)
    verify(data_dir)


def verify(data_dir):
    root = Path(data_dir)
    for split, total in EXPECTED.items():
        expected = total // len(CLASSES)
        for label in CLASSES:
            actual = len(list((root / split / label).glob("*.png")))
            if actual != expected:
                raise ValueError(f"{split}/{label}: expected {expected}, got {actual}")


def load_split(data_dir, split):
    root = Path(data_dir) / split
    paths = []
    labels = []
    for index, label in enumerate(CLASSES):
        for path in sorted((root / label).glob("*.png")):
            paths.append(str(path))
            labels.append(index)
    return paths, np.asarray(labels, dtype=np.int64)


def limit_split(paths, labels, size, seed):
    if size is None or size >= len(paths):
        return paths, labels
    indices = np.arange(len(paths))
    selected, _ = train_test_split(indices, train_size=size, stratify=labels, random_state=seed)
    selected = np.sort(selected)
    return [paths[index] for index in selected], labels[selected]


def search_indices(labels, size, seed):
    if size >= len(labels):
        return np.arange(len(labels))
    indices = np.arange(len(labels))
    selected, _ = train_test_split(indices, train_size=size, stratify=labels, random_state=seed)
    return np.sort(selected)
