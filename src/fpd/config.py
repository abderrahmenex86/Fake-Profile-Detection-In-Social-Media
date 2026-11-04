from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("seed", 1337)
    config.setdefault("backend", "cpu")
    config.setdefault("budget", 60)
    config.setdefault("seeds", [0, 1, 2])
    config.setdefault("search_train_size", 2000)
    config.setdefault("data_dir", str(Path.home() / "downloads/datasets/x_fake_profile_detection"))
    config["data_dir"] = str(Path(config["data_dir"]).expanduser())
    config.setdefault("cache_dir", str(Path(config["data_dir"]) / "features"))
    config["cache_dir"] = str(Path(config["cache_dir"]).expanduser())
    config.setdefault("results_dir", "results")
    config["results_dir"] = str(Path(config["results_dir"]).expanduser())
    config.setdefault("mlp_epochs", 30)
    config.setdefault("mlp_batch_size", 128)
    config.setdefault("cnn_epochs", 20)
    config.setdefault("cnn_batch_size", 64)
    config.setdefault("cnn_lr", 1e-3)
    config.setdefault("cnn_weight_decay", 1e-4)
    return config


def feature_label(feature):
    if feature.get("pca"):
        return f"{feature['name']}_pca{feature['pca_components']}"
    return f"{feature['name']}_nopca"


def ensure_dirs(config):
    Path(config["cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["results_dir"], "plots").mkdir(parents=True, exist_ok=True)
