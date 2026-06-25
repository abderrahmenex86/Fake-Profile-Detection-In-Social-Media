import argparse
import datetime
import json
import os

from sklearn.utils import shuffle

from src.dataset import load_image_paths
from src.factory import build_extractor
from src.infer import run_inference
from src.optimize import run_optimization
from src.tester import run_test
from src.trainer import run_training


class MockArgs:
    def __init__(self, original_args, model, feature, pca, optimizer):
        self.__dict__.update(original_args.__dict__)
        self.model = model
        self.feature = feature
        self.pca = pca
        self.optimizer = optimizer
        self.use_cuml = getattr(original_args, "use_cuml", False)


def get_run_dir(model, feature, pca, optimizer, is_training=False, run_dir_override=None):
    if not is_training and run_dir_override:
        return run_dir_override

    if is_training:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pca_str = "_pca" if pca else "_nopca"
        opt_str = f"_{optimizer }" if optimizer and optimizer != "none" else ""
        suffix = f"{model }_{feature }{pca_str }{opt_str }"
        run_dir = os.path.join("artifacts", f"{now }_{suffix }")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    else:
        if not os.path.exists("artifacts"):
            raise ValueError("No artifacts found.")
        dirs = [
            os.path.join("artifacts", d) for d in os.listdir("artifacts") if os.path.isdir(os.path.join("artifacts", d))
        ]
        if not dirs:
            raise ValueError("No previous run folders found.")
        return max(dirs, key=os.path.getmtime)


def is_already_completed(model, feature, pca, optimizer, mode):
    if not os.path.exists("artifacts"):
        return False

    for run_dir in os.listdir("artifacts"):
        dir_path = os.path.join("artifacts", run_dir)
        if not os.path.isdir(dir_path):
            continue

        hp_file = os.path.join(dir_path, "hyperparameters.json")
        summary_file = os.path.join(dir_path, "optuna_summary.json")
        history_file = os.path.join(dir_path, f"{model }_history.json")

        if os.path.exists(hp_file):
            try:
                with open(hp_file, "r") as f:
                    hp = json.load(f)

                match = hp.get("model") == model and hp.get("feature") == feature and hp.get("pca") == pca

                if mode == "optimize":
                    match = match and hp.get("optimizer") == optimizer and os.path.exists(summary_file)
                elif mode == "train":
                    match = match and os.path.exists(history_file)

                if match:
                    return True
            except Exception:
                continue
    return False


def main():
    parser = argparse.ArgumentParser(description="Master Thesis: Fake Profile Detection Sweep Engine")
    parser.add_argument("--mode", type=str, required=True, choices=["test", "train", "optimize", "infer"])

    parser.add_argument("--model", type=str, default="rf", choices=["nb", "lr", "svm", "rf", "xgb", "all"])
    parser.add_argument("--feature", type=str, default="raw", choices=["raw", "hog", "sift", "all"])
    parser.add_argument("--pca", action="store_true", help="Apply PCA to the chosen feature")
    parser.add_argument("--pca_components", type=int, default=500)
    parser.add_argument("--optimizer", type=str, default="ga", choices=["ga", "sa", "pso", "tow", "all"])

    parser.add_argument("--use_cuml", action="store_true", help="Accelerate SVM, RF, and LR using NVIDIA RAPIDS cuML")

    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    if args.mode == "infer":
        run_dir = get_run_dir(args.model, args.feature, args.pca, args.optimizer, False, args.run_dir)
        run_inference(run_dir)
        return

    models_to_run = ["nb", "lr", "svm", "rf", "xgb"] if args.model == "all" else [args.model]

    if args.feature == "all":

        features_to_run = [
            ("raw", False),
            ("raw", True),
            ("hog", False),
            ("hog", True),
            ("sift", False),
            ("sift", True),
        ]
    else:
        features_to_run = [(args.feature, args.pca)]

    optimizers_to_run = ["ga", "sa", "pso", "tow"] if args.optimizer == "all" else [args.optimizer]

    if args.mode == "test":
        optimizers_to_run = ["none"]

    if args.mode in ["train", "optimize"]:
        print("[INFO] Loading image paths...")
        train_paths, y_train = load_image_paths("dataset", "train")
        val_paths, y_val = load_image_paths("dataset", "validation")
        train_paths, y_train = shuffle(train_paths, y_train, random_state=42)
    else:
        train_paths, y_train, val_paths, y_val = None, None, None, None

    total_runs = len(models_to_run) * len(features_to_run) * len(optimizers_to_run)
    run_idx = 1

    for model in models_to_run:
        for feature, pca in features_to_run:
            for optimizer in optimizers_to_run:

                if total_runs > 1:
                    print(f"\n{'='*80 }")
                    print(
                        f"[GRID SWEEP {run_idx }/{total_runs }] Model: {model .upper ()} | Feature: {feature .upper ()}"
                        + ("+PCA" if pca else "")
                        + f" | Opt: {optimizer .upper ()}"
                    )
                    print(f"{'='*80 }")

                if total_runs > 1 and is_already_completed(model, feature, pca, optimizer, args.mode):
                    print(f"[INFO] Configuration detected as already completed. Safely skipping.")
                    run_idx += 1
                    continue

                run_args = MockArgs(args, model, feature, pca, optimizer)

                if args.mode == "test":
                    run_test(run_args)
                    run_idx += 1
                    continue

                run_dir = get_run_dir(model, feature, pca, optimizer, is_training=True)
                with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
                    json.dump(run_args.__dict__, f, indent=4)

                run_train_paths, run_y_train = train_paths, y_train
                if args.mode == "optimize":
                    run_train_paths, run_y_train = train_paths[:1000], y_train[:1000]

                extractor = build_extractor(run_args)
                X_train = extractor.fit_transform(run_train_paths)
                X_val = extractor.transform(val_paths)

                best_params = None
                if args.mode == "optimize":
                    best_params = run_optimization(run_args, X_train, run_y_train, run_dir)

                run_training(run_args, X_train, run_y_train, X_val, y_val, extractor, run_dir, best_params)
                run_idx += 1


if __name__ == "__main__":
    main()
