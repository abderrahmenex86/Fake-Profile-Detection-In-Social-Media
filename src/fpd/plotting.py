import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _read(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def _convergence(runs, output):
    runs = runs.copy()
    runs["history"] = runs["history"].apply(json.loads)
    for feature, group in runs.groupby("feature"):
        models = sorted(group["model"].unique())
        figure, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.2), sharey=True)
        if len(models) == 1:
            axes = [axes]
        for axis, model in zip(axes, models):
            subset = group[group["model"] == model]
            for optimizer, optimizer_group in subset.groupby("optimizer"):
                histories = [history for history in optimizer_group["history"] if len(history) > 0]
                if not histories:
                    continue
                length = min(len(history) for history in histories)
                curve = np.mean([history[:length] for history in histories], axis=0)
                axis.plot(range(1, length + 1), curve, label=optimizer)
            axis.set_title(model)
            axis.set_xlabel("evaluation")
        axes[0].set_ylabel("best validation macro F1")
        axes[-1].legend(fontsize=7, loc="lower right")
        figure.tight_layout()
        figure.savefig(output / f"fig_convergence_{feature}.png", dpi=160)
        plt.close(figure)


def _optimizer_box(runs, output):
    figure, axis = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=runs, x="optimizer", y="best_val_macro_f1", ax=axis)
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("Optimizer comparison")
    figure.tight_layout()
    figure.savefig(output / "fig_optimizer_box.png", dpi=160)
    plt.close(figure)


def _final_heatmap(final, output):
    pivot = final.pivot_table(index="feature", columns="model", values="test_macro_f1", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=axis)
    axis.set_title("Final test macro F1")
    figure.tight_layout()
    figure.savefig(output / "fig_final_heatmap.png", dpi=160)
    plt.close(figure)


def _pca_effect(final, output):
    rows = []
    for _, row in final.iterrows():
        if "pca" not in row["feature"]:
            continue
        base = row["feature"].split("_pca")[0]
        baseline = final[(final["feature"] == f"{base}_nopca") & (final["model"] == row["model"])]
        if baseline.empty:
            continue
        rows.append(
            {
                "base": base,
                "model": row["model"],
                "delta": row["test_macro_f1"] - baseline.iloc[0]["test_macro_f1"],
            }
        )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.barplot(data=frame, x="model", y="delta", hue="base", ax=axis)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("PCA minus no-PCA")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output / "fig_pca_effect.png", dpi=160)
    plt.close(figure)


def _feature_effect(final, output):
    subset = final[final["feature"].isin(["hog_nopca", "sift_nopca", "hog_pca1000", "sift_pca1000"])]
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4))
    sns.barplot(data=subset, x="model", y="test_macro_f1", hue="feature", ax=axis)
    axis.set_title("HOG vs SIFT")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output / "fig_hog_vs_sift.png", dpi=160)
    plt.close(figure)


def _cost(final, output):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(final["train_time_s"], final["test_macro_f1"])
    axes[0].set_xlabel("train time (s)")
    axes[0].set_ylabel("test macro F1")
    axes[1].scatter(final["inference_ms_batch1"], final["test_macro_f1"])
    axes[1].set_xlabel("inference batch 1 (ms)")
    axes[2].scatter(final["model_size_mb"], final["test_macro_f1"])
    axes[2].set_xlabel("model size (MB)")
    for axis in axes:
        axis.set_ylabel("test macro F1")
    figure.tight_layout()
    figure.savefig(output / "fig_cost_accuracy.png", dpi=160)
    plt.close(figure)


def _cnn_effect(final, cnn, output):
    if cnn is None:
        return
    best = final.sort_values("test_macro_f1", ascending=False).head(6).copy()
    best["label"] = best["feature"] + " / " + best["model"]
    cnn_value = cnn["test_macro_f1"].mean()
    labels = list(best["label"]) + ["CNN"]
    values = list(best["test_macro_f1"]) + [cnn_value]
    figure, axis = plt.subplots(figsize=(9, 4))
    sns.barplot(x=labels, y=values, ax=axis)
    axis.set_ylabel("test macro F1")
    axis.set_title("CNN vs classical and MLP")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output / "fig_cnn_comparison.png", dpi=160)
    plt.close(figure)


def _ablation(ablation, output):
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ablation, x="variant", y="test_macro_f1", hue="model", ax=axis)
    axis.set_title("Template ablation")
    figure.tight_layout()
    figure.savefig(output / "fig_template_ablation.png", dpi=160)
    plt.close(figure)


def _confusion(final, output):
    if final.empty:
        return
    best = final.sort_values("test_macro_f1", ascending=False).iloc[0]
    matrix = json.loads(best["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(f"{best['feature']} / {best['model']}")
    figure.tight_layout()
    figure.savefig(output / "fig_best_confusion.png", dpi=160)
    plt.close(figure)


def run(config):
    results = Path(config["results_dir"])
    output = results / "plots"
    output.mkdir(parents=True, exist_ok=True)
    runs = _read(results / "runs.csv")
    final = _read(results / "final.csv")
    cnn = _read(results / "cnn.csv")
    ablation = _read(results / "ablation.csv")
    if runs is not None and not runs.empty:
        _convergence(runs, output)
        _optimizer_box(runs, output)
    if final is not None and not final.empty:
        _final_heatmap(final, output)
        _pca_effect(final, output)
        _feature_effect(final, output)
        _cost(final, output)
        _confusion(final, output)
        _cnn_effect(final, cnn, output)
    if ablation is not None and not ablation.empty:
        _ablation(ablation, output)
    print(f"[plot] wrote {output}", flush=True)
