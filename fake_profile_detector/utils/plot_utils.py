import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_accuracy_comparison(results_df, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y="Accuracy", hue="Optimizer")
    plt.title("Validation Accuracy by Model and Optimizer")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.legend(title="Optimizer", loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved accuracy plot to {save_path}")
    else:
        plt.show()


def plot_time_vs_accuracy(results_df, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results_df, x="Time", y="Accuracy", hue="Model", style="Optimizer", s=100
    )
    plt.title("Time vs Accuracy by Model and Optimizer")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved time vs accuracy plot to {save_path}")
    else:
        plt.show()


def plot_convergence_curve(history, save_path=None, title="Convergence Curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(history, marker="o")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Accuracy)")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved convergence curve to {save_path}")
    else:
        plt.show()
