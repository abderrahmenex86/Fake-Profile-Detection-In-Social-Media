import argparse
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(prog="fpd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument(
        "--data-dir",
        default=str(Path.home() / "downloads/datasets/x_fake_profile_detection"),
    )

    defaults = {
        "features": "configs/stage_a.yaml",
        "smoke": "configs/smoke.yaml",
        "benchmark": "configs/stage_a.yaml",
        "final": "configs/final.yaml",
        "cnn": "configs/cnn.yaml",
        "ablation": "configs/ablation.yaml",
        "plot": "configs/final.yaml",
    }
    for name, default in defaults.items():
        command = subparsers.add_parser(name)
        command.add_argument("--config", default=default)
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "download":
        from .data import download

        download(args.data_dir)
        return

    from .config import ensure_dirs, load_config

    config = load_config(args.config)
    ensure_dirs(config)

    if args.command in {"features", "smoke", "benchmark"}:
        from .benchmark import features_only, run

        if args.command == "features":
            features_only(config)
        else:
            run(config)
    elif args.command == "final":
        from .final_eval import run

        run(config)
    elif args.command == "cnn":
        from .cnn import run

        run(config)
    elif args.command == "ablation":
        from .ablation import run

        run(config)
    elif args.command == "plot":
        from .plotting import run

        run(config)


if __name__ == "__main__":
    main()
