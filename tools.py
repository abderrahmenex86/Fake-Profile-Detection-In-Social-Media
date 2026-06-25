import argparse
import json
import os

import matplotlib.pyplot as plt

from src.helpers import download_and_prepare_dataset

os.makedirs("docs/figs", exist_ok=True)


def plot_analytics():
    runs = []
    if not os.path.exists("artifacts"):
        return

    for d in os.listdir("artifacts"):
        run_path = os.path.join("artifacts", d)
        if not os.path.isdir(run_path):
            continue

        optuna_file = os.path.join(run_path, "optuna_summary.json")
        if os.path.exists(optuna_file):
            with open(optuna_file, "r") as f:
                runs.append(json.load(f))

    if not runs:
        print("[ERROR] No optimization data found.")
        return

    plt.figure(figsize=(10, 6))
    for r in runs:
        label = f"{r['model'].upper()} ({r['optimizer'].upper()})"
        epochs = range(1, len(r["history"]) + 1)
        plt.plot(epochs, r["history"], marker="o", label=label, linewidth=2)

    plt.title("Metaheuristic Optimization Convergence", fontsize=14, fontweight="bold")
    plt.xlabel("Generation / Iteration", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/figs/optimization_convergence.png", dpi=300)
    print("[SUCCESS] Saved docs/figs/optimization_convergence.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["download", "plot"])
    args = parser.parse_args()

    if args.mode == "download":
        download_and_prepare_dataset()
    elif args.mode == "plot":
        plot_analytics()


if __name__ == "__main__":
    main()
