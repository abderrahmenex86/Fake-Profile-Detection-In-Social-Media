import gc
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psutil

from fake_profile_detector.configs.general import FEATURES_DIR, SAVE_DIR
from fake_profile_detector.extractors.evaluate_features import \
    evaluate_features
from fake_profile_detector.extractors.FeatureExtractor import FeatureExtractor
from fake_profile_detector.extractors.load_data import load_data


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def save_features(X, y_valid, valid_indices, method_name, timestamp):
    safe_method_name = method_name.replace(" ", "_").replace("+", "plus")

    features_save_dir = os.path.join(FEATURES_DIR, "x")
    os.makedirs(features_save_dir, exist_ok=True)

    feature_data = {
        "features": X,
        "labels": y_valid,
        "valid_indices": valid_indices,
        "method_name": method_name,
        "feature_shape": X.shape,
        "timestamp": timestamp,
    }

    feature_file = os.path.join(
        features_save_dir, f"features_{safe_method_name}_{timestamp}.joblib"
    )
    joblib.dump(feature_data, feature_file)

    print(f"Features saved to: {feature_file}")
    return feature_file


def main():
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")

    image_paths, labels = load_data()
    print(f"Loaded {len(image_paths)} images with labels.")
    print(f"Memory usage after loading paths: {get_memory_usage():.1f} MB")

    feature_methods = [
        ("Raw Pixels", "raw_pixel", False, 500),  # Updated to batch size 500
        ("Raw Pixels + PCA", "raw_pixel", True, 500),
        ("HOG", "hog", False, 500),  # Updated to batch size 500
        ("HOG + PCA", "hog", True, 500),
        ("SIFT", "sift", False, 500),
        ("SIFT + PCA", "sift", True, 500),
        # SURF removed - not available in this OpenCV build
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    saved_features = []

    for method_name, feature_type, use_pca, batch_size in feature_methods:
        print(f"\n{'='*50}")
        print(f"Processing: {method_name}")
        print(f"{'='*50}")
        print(f"Memory usage before extraction: {get_memory_usage():.1f} MB")

        extractor = FeatureExtractor(
            feature_type=feature_type,
            apply_pca=use_pca,
            batch_size=batch_size,
        )

        start_time = time.time()
        X, valid_indices = extractor.fit_transform(image_paths)
        extraction_time = time.time() - start_time

        y_valid = labels[valid_indices]

        print(f"Feature extraction completed in {extraction_time:.2f}s")
        print(f"Feature shape: {X.shape}")
        print(f"Memory usage after extraction: {get_memory_usage():.1f} MB")

        feature_file = save_features(X, y_valid, valid_indices, method_name, timestamp)
        saved_features.append(
            {
                "method_name": method_name,
                "feature_file": feature_file,
                "feature_shape": X.shape,
            }
        )

        evaluation_results = evaluate_features(X, y_valid, method_name, extractor)

        for clf_name, metrics in evaluation_results.items():
            all_results.append(
                {
                    "Feature_Method": method_name,
                    "Classifier": clf_name,
                    "Accuracy": metrics["accuracy"],
                    "Training_Time": metrics["train_time"],
                    "Feature_Dimension": metrics["feature_dim"],
                    "Extraction_Time": extraction_time,
                }
            )

        del X, y_valid, extractor
        gc.collect()
        print(f"Memory usage after cleanup: {get_memory_usage():.1f} MB")

    results_df = pd.DataFrame(all_results)

    results_file = os.path.join(SAVE_DIR, "logs", f"feature_comparison_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    features_summary = pd.DataFrame(saved_features)
    features_summary_file = os.path.join(
        SAVE_DIR, "logs", f"saved_features_summary_{timestamp}.csv"
    )
    features_summary.to_csv(features_summary_file, index=False)
    print(f"Features summary saved to: {features_summary_file}")

    return results_df


if __name__ == "__main__":
    main()
