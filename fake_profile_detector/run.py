import argparse
import os
import time
from datetime import datetime

import pandas as pd

from .configs.general import SAVE_DIR
from .models import logistic_regression, naive_bayes, random_forest, svm, xgb
from .optimizers import ga, pso, tow
from .utils.data_loader import load_data
from .utils.metrics import compute_metrics


def run_experiments(dataset, selected_models=None, selected_optimizers=None):
    (
        X_final_train,
        y_final_train,
        X_opt_train,
        y_opt_train,
        X_opt_val,
        y_opt_val,
        X_final_test,
        y_final_test,
    ) = load_data(dataset)

    all_models = [
        ("NaiveBayes", naive_bayes),
        ("LogisticRegression", logistic_regression),
        ("SVM", svm),
        ("RandomForest", random_forest),
        ("XGBoost", xgb),
    ]
    all_optimizers = [
        ("Baseline", None),
        ("GA", ga.optimize),
        ("PSO", pso.optimize),
        ("TOW", tow.optimize),
    ]

    if selected_models:
        models = [
            (name, module) for name, module in all_models if name in selected_models
        ]
        if not models:
            print(
                f"Warning: None of the specified models {selected_models} were found. Using all models."
            )
            models = all_models
    else:
        models = all_models

    if selected_optimizers:
        if "Baseline" not in selected_optimizers:
            selected_optimizers.append("Baseline")
        optimizers = [
            (name, opt) for name, opt in all_optimizers if name in selected_optimizers
        ]
        if not optimizers:
            print(
                f"Warning: None of the specified optimizers {selected_optimizers} were found. Using all optimizers."
            )
            optimizers = all_optimizers
    else:
        optimizers = all_optimizers

    results = []

    for model_name, model_module in models:

        print(f"Running {model_name} with default parameters, no optimizer...")

        time_start = time.time()
        baseline_model = model_module.create_model(model_module.default_params())
        baseline_model.fit(X_opt_train, y_opt_train)
        time_taken = time.time()
        time_taken = time_taken - time_start

        print(
            f"Running {model_name} with default parameters, no optimizer... done in {time_taken:.2f} seconds"
        )

        for split_name, X, y in [
            ("Train", X_opt_train, y_opt_train),
            ("Test", X_final_test, y_final_test),
        ]:
            y_pred = baseline_model.predict(X)
            y_prob = (
                baseline_model.predict_proba(X)
                if hasattr(baseline_model, "predict_proba")
                else None
            )
            mets = compute_metrics(y, y_pred, y_prob)
            results.append(
                {
                    "Model": model_name,
                    "Dataset": dataset,
                    "Optimizer": "Baseline",
                    "Split": split_name,
                    **mets,
                    "Time": time_taken,
                    "Params": model_module.default_params(),
                }
            )

        for opt_name, optimizer in optimizers[1:]:
            print(f"Running {model_name} with {opt_name} optimizer...")
            time_start = time.time()
            best_model, best_params, best_score = optimizer(
                model_module, X_opt_train, y_opt_train, X_opt_val, y_opt_val
            )
            time_end = time.time()
            time_taken = time_end - time_start
            print(
                f"Running {model_name} with {opt_name} optimizer... done in {time_taken:.2f} seconds"
            )
            for split_name, X, y in [
                ("Train", X_opt_train, y_opt_train),
                ("Test", X_final_test, y_final_test),
            ]:
                y_pred = best_model.predict(X)
                y_prob = (
                    best_model.predict_proba(X)
                    if hasattr(best_model, "predict_proba")
                    else None
                )
                mets = compute_metrics(y, y_pred, y_prob)
                results.append(
                    {
                        "Model": model_name,
                        "Dataset": dataset,
                        "Optimizer": opt_name,
                        "Split": split_name,
                        **mets,
                        "Time": time_taken,
                        "Params": best_params,
                    }
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(SAVE_DIR, "logs", f"experiment_results_{timestamp}.csv")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Experiments completed. Results saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fake profile detection experiments"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="x",
        help="Dataset to use (default: instagram)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "SVM",
            "SGD",
            "RandomForest",
            "XGBoost",
            "LogisticRegression",
            "AdaBoost",
            "KNN",
            "MLP",
            "NaiveBayes",
        ],
        help="Models to run (default: all models)",
    )

    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["Baseline", "GA", "PSO", "SA", "TOW"],
        help="Optimizers to run (default: all optimizers)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        dataset=args.dataset,
        selected_models=args.models,
        selected_optimizers=args.optimizers,
    )
