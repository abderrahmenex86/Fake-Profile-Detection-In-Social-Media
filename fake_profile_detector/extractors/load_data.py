import os

import numpy as np
from tqdm import tqdm

from fake_profile_detector.configs.general import BASE_DIR


def load_data():
    base_dir = os.path.join(
        BASE_DIR,
        "x",
    )
    os.makedirs(base_dir, exist_ok=True)

    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading image files...")
    image_files = [f for f in os.listdir(base_dir) if f.endswith(".png")]
    
    print("Processing labels...")
    labels = []
    for image in tqdm(image_files, desc="Processing image labels"):
        label = image.split("_")[1]
        labels.append(label)
    
    labels = np.array(labels)
    print(f"Loaded {len(image_files)} images with {len(np.unique(labels))} unique labels")

    return image_files, labels
