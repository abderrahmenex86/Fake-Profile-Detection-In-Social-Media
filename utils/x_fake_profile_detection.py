import os

import numpy as np
from datasets import load_dataset
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from tqdm.auto import tqdm


def extract_hog_features(
    image_path, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)
):
    try:
        img = imread(image_path)

        if img.ndim == 3 and img.shape[2] == 4:
            from skimage.color import rgba2rgb

            img = rgba2rgb(img)

        img_gray = rgb2gray(img)

        features = hog(
            img_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True,
            channel_axis=None,
        )
        return features
    except Exception as _:
        return None


def prepare(recompute=False):
    # data/x_fake_profile_detection
    # data/x_fake_profile_detection/cache
    dataset = load_dataset("drveronika/x_fake_profile_detection")

    base_dir = os.path.join(
        os.environ["HOME"],
        "master",
        "fake-detector",
        "data",
        "x_fake_profile_detection",
    )
    os.makedirs(base_dir, exist_ok=True)

    # X = []
    # y = []
    X = None
    y = None

    _hog_params = [
        {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
        },
        {
            "orientations": 9,
            "pixels_per_cell": (16, 16),
            "cells_per_block": (2, 2),
        },
        {
            "orientations": 9,
            "pixels_per_cell": (32, 32),
            "cells_per_block": (2, 2),
        },
    ]
    for hog_params in _hog_params:
        print(f"Extracting features with HOG parameters: {hog_params}")
        config_name = f"hog_o{hog_params['orientations']}_p{hog_params['pixels_per_cell'][0]}x{hog_params['pixels_per_cell'][1]}_c{hog_params['cells_per_block'][0]}x{hog_params['cells_per_block'][1]}"

        cache_dir = os.path.join(base_dir, "cache")
        cache_file_x = os.path.join(cache_dir, f"features_X_{config_name}.npy")
        cache_file_y = os.path.join(cache_dir, f"labels_y_{config_name}.npy")
        X = None
        y = None

        os.makedirs(cache_dir, exist_ok=True)

        if (
            not recompute
            and os.path.exists(cache_file_x)
            and os.path.exists(cache_file_y)
        ):
            print(f"Loading cached features for config: {config_name} from {cache_dir}")
            X = np.load(cache_file_x)
            y = np.load(cache_file_y)
        else:
            all_features = []
            all_labels = []
            for split_name, split_data in dataset.items():
                for i, item in enumerate(
                    tqdm(split_data, desc=f"Saving {split_name} images")
                ):
                    img = item["image"]
                    label = item["label"]

                    filepath = os.path.join(base_dir, f"{i:05}_{label}.png")

                    if not os.path.exists(filepath):
                        img.save(filepath)

                    features = extract_hog_features(filepath)
                    all_features.append(features)
                    all_labels.append(label)

            if not all_features:
                raise ValueError(
                    "No features were extracted. Check image paths, extensions, and HOG parameters."
                )

            X = np.array(all_features)
            y = np.array(all_labels)

            print(f"Saving extracted features to {cache_dir} for config: {config_name}")
            np.save(cache_file_x, X)
            np.save(cache_file_y, y)

    # return X, y


def load_data(preprocess=True):
    if preprocess:
        X, y = prepare()
    else:
        base_dir = os.path.join(
            os.environ["HOME"],
            "master",
            "fake-detector",
            "data",
            "x_fake_profile_detection",
        )
        filenames = os.listdir(base_dir)
        X = filenames
        y = [filename.split(".")[0].split("_")[1] for filename in filenames]

    return X, y
