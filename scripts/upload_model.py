import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. username/fake-profile-hog-xgb")
    parser.add_argument("--folder", default="models/hog_xgb")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Upload HOG + XGBoost fake-profile classifier")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not (folder / "model.ubj").exists():
        raise FileNotFoundError(f"{folder}/model.ubj not found; run scripts/export_best_model.py first")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
        ignore_patterns=["__pycache__/*", "*.pyc", ".ipynb_checkpoints/*"],
    )
    print(f"https://huggingface.co/models/{args.repo_id}")


if __name__ == "__main__":
    main()
