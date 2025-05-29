import ast
import glob
import os
import pickle

import pandas as pd

from fake_profile_detector.configs.general import SAVE_DIR

from .configs.general import SAVE_DIR
from .models import logistic_regression, naive_bayes, random_forest, svm, xgb
from .utils.data_loader import load_data
from .utils.metrics import compute_metrics, print_classification_report


def train():
    (
        X_final_train,
        y_final_train,
        _,
        _,
        _,
        _,
        X_final_test,
        y_final_test,
    ) = load_data("x")

    logs_dir = os.path.join(SAVE_DIR, "logs")
    models_dir = os.path.join(SAVE_DIR, "models")
    log_files = glob.glob(os.path.join(logs_dir, "experiment_results_*.csv"))
    latest_file = max(log_files, key=os.path.getmtime)

    results_df = pd.read_csv(latest_file)
    tow_results = results_df[
        (results_df["Optimizer"] == "TOW") & (results_df["Split"] == "Test")
    ]

    model_modules = {
        "NaiveBayes": naive_bayes,
        "LogisticRegression": logistic_regression,
        "SVM": svm,
        "RandomForest": random_forest,
        "XGBoost": xgb,
    }

    trained_models = {}

    for _, row in tow_results.iterrows():
        model_name = row["Model"]
        params = ast.literal_eval(row["Params"])

        print(f"\n=== Training {model_name} with optimized parameters ===")
        print(f"Parameters: {params}")

        if model_name in model_modules:
            module = model_modules[model_name]

            model = module.create_model(params)
            print(f"Training {model_name} on full training dataset...")
            model.fit(X_final_train, y_final_train)

            y_pred = model.predict(X_final_test)
            y_prob = (
                model.predict_proba(X_final_test)
                if hasattr(model, "predict_proba")
                else None
            )

            print(f"\nTest set evaluation for {model_name}:")
            metrics = compute_metrics(y_final_test, y_pred, y_prob)
            for metric, value in metrics.items():
                print(f"{metric}: {value}")

            print("\nClassification Report:")
            print_classification_report(y_final_test, y_pred)

            model_path = os.path.join(models_dir, f"{model_name.lower()}_tow.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved to {model_path}")

            trained_models[model_name] = model
        else:
            print(f"Warning: No module found for model {model_name}")

    print("\nAll models trained and saved successfully!")
    return trained_models


if __name__ == "__main__":
    train()
