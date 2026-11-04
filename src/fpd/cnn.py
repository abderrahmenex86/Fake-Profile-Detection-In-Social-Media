import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ensure_dirs
from .data import CLASSES, limit_split, load_split
from .tracking import (
    append_row,
    metrics,
    model_size_mb,
    param_count,
    reset_torch_peak,
    torch_peak_mb,
    write_metadata,
)


class ProfileDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB").resize((150, 200))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return torch.from_numpy(array.transpose(2, 0, 1)), int(self.labels[index])


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, features):
        return self.head(self.features(features))


def _evaluate(model, loader, device):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            predictions.append(model(features).argmax(dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def _tensor(path):
    image = Image.open(path).convert("RGB").resize((150, 200))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1))


def _inference_ms(model, paths, device, repeats=10):
    model.eval()
    single = _tensor(paths[0]).unsqueeze(0).to(device)
    batch = torch.stack([_tensor(path) for path in paths[: min(32, len(paths))]]).to(device)
    values_single = []
    values_batch = []
    with torch.no_grad():
        model(single)
        model(batch)
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(single)
            if device == "cuda":
                torch.cuda.synchronize()
            values_single.append((time.perf_counter() - start) * 1000)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            values_batch.append((time.perf_counter() - start) * 1000)
    return float(np.median(values_single)), float(np.median(values_batch))


def _run_seed(config, seed, train_paths, train_y, val_paths, val_y, test_paths, test_y):
    device = "cuda" if config["backend"] == "gpu" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    reset_torch_peak()
    model = SmallCNN().to(device)
    train_loader = DataLoader(
        ProfileDataset(train_paths, train_y),
        batch_size=config["cnn_batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ProfileDataset(val_paths, val_y),
        batch_size=config["cnn_batch_size"],
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ProfileDataset(test_paths, test_y),
        batch_size=config["cnn_batch_size"],
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["cnn_lr"],
        weight_decay=config["cnn_weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["cnn_epochs"])
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_val = -np.inf
    history = []
    start = time.perf_counter()
    for _ in range(config["cnn_epochs"]):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        targets, predictions = _evaluate(model, val_loader, device)
        value = metrics(targets, predictions, CLASSES)["macro_f1"]
        history.append(float(value))
        if value > best_val:
            best_val = value
            best_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    targets, predictions = _evaluate(model, test_loader, device)
    values = metrics(targets, predictions, CLASSES)
    inference_1, inference_32 = _inference_ms(model, test_paths, device)
    append_row(
        Path(config["results_dir"]) / "cnn.csv",
        {
            "seed": seed,
            "epochs": config["cnn_epochs"],
            "batch_size": config["cnn_batch_size"],
            "lr": config["cnn_lr"],
            "weight_decay": config["cnn_weight_decay"],
            "train_size": len(train_y),
            "val_macro_f1": best_val,
            "test_accuracy": values["accuracy"],
            "test_macro_f1": values["macro_f1"],
            "test_precision": values["macro_precision"],
            "test_recall": values["macro_recall"],
            "per_class": values["classification_report"],
            "confusion_matrix": json.dumps(values["confusion_matrix"]),
            "train_time_s": train_time,
            "inference_ms_batch1": inference_1,
            "inference_ms_batch32": inference_32,
            "model_size_mb": model_size_mb(model),
            "param_count": param_count(model),
            "gpu_mem_mb": torch_peak_mb(),
            "backend": config["backend"],
            "history": json.dumps(history),
        },
    )
    print(f"[cnn] seed={seed} val={best_val:.4f} test={values['macro_f1']:.4f}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(config):
    ensure_dirs(config)
    write_metadata(Path(config["results_dir"]) / "metadata.json", config)
    train_paths, train_y = load_split(config["data_dir"], "train")
    val_paths, val_y = load_split(config["data_dir"], "validation")
    test_paths, test_y = load_split(config["data_dir"], "test")
    train_paths, train_y = limit_split(train_paths, train_y, config.get("limit_train"), config["seed"])
    val_paths, val_y = limit_split(val_paths, val_y, config.get("limit_val"), config["seed"])
    test_paths, test_y = limit_split(test_paths, test_y, config.get("limit_test"), config["seed"])
    for seed in config["seeds"]:
        _run_seed(config, seed, train_paths, train_y, val_paths, val_y, test_paths, test_y)
