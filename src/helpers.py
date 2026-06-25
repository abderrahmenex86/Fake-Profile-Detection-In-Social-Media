import os
import shutil
import zipfile

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm


def download_and_prepare_dataset(dataset_dir="dataset"):
    print(f"[INFO] Bypassing HF datasets parser. Downloading raw ZIPs directly to {dataset_dir }...")
    os.makedirs(dataset_dir, exist_ok=True)

    repo_id = "drveronika/x_fake_profile_detection"
    splits = ["train", "validation", "test"]
    valid_labels = ["bot", "cyborg", "real", "verified"]

    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"[INFO] Downloading {split }.zip...")
        zip_path = hf_hub_download(repo_id=repo_id, filename=f"{split }.zip", repo_type="dataset")

        temp_extract_dir = os.path.join(dataset_dir, f"temp_{split }")
        print(f"[INFO] Extracting {split }.zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        print(f"[INFO] Formatting {split } images into clean folders...")
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    inferred_label = None
                    for label in valid_labels:
                        if label in root.lower().split(os.sep):
                            inferred_label = label.upper()
                            break

                    if inferred_label:
                        target_class_dir = os.path.join(split_dir, inferred_label)
                        os.makedirs(target_class_dir, exist_ok=True)

                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(target_class_dir, f"{len (os .listdir (target_class_dir ))}.png")

                        try:
                            img = Image.open(src_path).convert("RGB")
                            img.save(dst_path)
                        except Exception as e:
                            pass

        shutil.rmtree(temp_extract_dir)

    print("[SUCCESS] Dataset downloaded and strictly formatted.")


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics.update({"precision": float(p), "recall": float(r), "f1": float(f)})

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except:
            metrics["roc_auc"] = 0.0
    return metrics
