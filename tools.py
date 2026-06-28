import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.helpers import download_and_prepare_dataset

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 12, "figure.autolayout": True})

os.makedirs("docs/figs", exist_ok=True)


def load_all_runs():
    runs = []
    if not os.path.exists("artifacts"):
        print("[ERROR] No artifacts directory found.")
        return pd.DataFrame()

    for d in os.listdir("artifacts"):
        run_path = os.path.join("artifacts", d)
        if not os.path.isdir(run_path):
            continue

        hp_file = os.path.join(run_path, "hyperparameters.json")
        opt_file = os.path.join(run_path, "optuna_summary.json")

        if not os.path.exists(hp_file) or not os.path.exists(opt_file):
            continue

        try:
            with open(hp_file, "r") as f:
                hp = json.load(f)
            with open(opt_file, "r") as f:
                opt = json.load(f)

            hist_file = os.path.join(run_path, f"{hp['model']}_history.json")
            val_metrics = {}
            if os.path.exists(hist_file):
                with open(hist_file, "r") as f:
                    val_metrics = json.load(f).get("val", {})

            feat_name = hp["feature"].upper()
            if hp["pca"]:
                feat_name += " + PCA"

            runs.append(
                {
                    "Run_ID": d,
                    "Model": hp["model"].upper(),
                    "Feature": feat_name,
                    "Base_Feature": hp["feature"].upper(),
                    "PCA": "Yes" if hp["pca"] else "No",
                    "Optimizer": hp["optimizer"].upper(),
                    "CV_Accuracy": opt.get("best_accuracy", 0.0),
                    "Val_Accuracy": val_metrics.get("accuracy", 0.0),
                    "Val_Precision": val_metrics.get("precision", 0.0),
                    "Val_Recall": val_metrics.get("recall", 0.0),
                    "Val_F1": val_metrics.get("f1", 0.0),
                    "History": opt.get("history", []),
                }
            )
        except Exception as e:
            print(f"[WARNING] Failed to parse {d}: {e}")

    return pd.DataFrame(runs)


def plot_heatmap(df):
    print("[INFO] Generating Global Performance Heatmap...")

    pivot = df.pivot_table(index="Model", columns="Feature", values="Val_Accuracy", aggfunc="max")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Validation Accuracy"})
    plt.title("Max Validation Accuracy: Model vs. Feature Configuration", fontweight="bold", pad=15)
    plt.ylabel("Classifier")
    plt.xlabel("Feature Extraction Method")
    plt.savefig("docs/figs/01_global_performance_heatmap.png", dpi=300)
    plt.close()


def plot_classifier_comparison(df):
    print("[INFO] Generating Classifier Robustness Comparison...")

    filtered_df = df[df["Val_Accuracy"] > 0.3]

    order = filtered_df.groupby("Model")["Val_Accuracy"].median().sort_values(ascending=False).index

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=filtered_df, x="Model", y="Val_Accuracy", order=order, palette="Set3", hue="Model", legend=False)
    plt.title("Classifier Robustness: Validation Accuracy Distribution", fontweight="bold", pad=15)
    plt.xlabel("Classifier Model")
    plt.ylabel("Validation Accuracy")
    plt.savefig("docs/figs/02_classifier_comparison.png", dpi=300)
    plt.close()


def plot_pca_impact(df):
    print("[INFO] Generating PCA Impact Analysis...")

    filtered_df = df[df["Val_Accuracy"] > 0.3]

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=filtered_df, x="Base_Feature", y="Val_Accuracy", hue="PCA", palette="Set2")
    plt.title("Impact of PCA Dimensionality Reduction on Accuracy", fontweight="bold", pad=15)
    plt.xlabel("Base Feature Extractor")
    plt.ylabel("Validation Accuracy")
    plt.legend(title="PCA Applied", loc="lower left")
    plt.savefig("docs/figs/03_pca_impact_analysis.png", dpi=300)
    plt.close()


def plot_convergence_race(df):
    print("[INFO] Generating Optimizer Convergence Race...")

    best_combo = None
    max_variance = -1

    grouped = df.groupby(["Model", "Feature"])
    for (model, feature), group in grouped:
        total_variance = 0
        for _, row in group.iterrows():
            hist = row["History"]
            if len(hist) > 1:
                total_variance += np.var(hist) + (max(hist) - min(hist))

        if total_variance > max_variance:
            max_variance = total_variance
            best_combo = (model, feature)

    if not best_combo:
        print("[WARNING] Could not find any active histories to plot.")
        return

    best_model, best_feat = best_combo
    subset = df[(df["Model"] == best_model) & (df["Feature"] == best_feat)]

    plt.figure(figsize=(10, 6))
    colors = {"GA": "tab:blue", "PSO": "tab:orange", "SA": "tab:red", "TOW": "tab:green"}

    for _, row in subset.iterrows():
        opt = row["Optimizer"]
        history = row["History"]
        epochs = range(1, len(history) + 1)
        plt.plot(
            epochs,
            history,
            marker="o",
            label=f"{opt} (Max: {max(history):.3f})",
            color=colors.get(opt, "black"),
            linewidth=2,
            markersize=6,
        )

    plt.title(f"Metaheuristic Convergence: {best_model} on {best_feat}", fontweight="bold", pad=15)
    plt.xlabel("Optimization Generation / Step")
    plt.ylabel("Cross-Validation Accuracy")
    plt.legend(title="Optimizer")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("docs/figs/04_optimizer_convergence_race.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["download", "plot"])
    args = parser.parse_args()

    if args.mode == "download":
        download_and_prepare_dataset()
    elif args.mode == "plot":
        df = load_all_runs()
        if df.empty:
            print("[ERROR] No data to plot. Check your artifacts directory.")
            return

        print(f"[INFO] Successfully loaded {len(df)} configurations.")
        plot_heatmap(df)
        plot_classifier_comparison(df)
        plot_pca_impact(df)
        plot_convergence_race(df)
        print("\n[SUCCESS] All upgraded thesis plots successfully generated in 'docs/figs/'!")


if __name__ == "__main__":
    main()
