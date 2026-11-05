import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fpd.data import CLASSES, load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="~/downloads/datasets/x_fake_profile_detection")
    parser.add_argument("--output", default="web/samples")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths, labels = load_split(Path(args.data_dir).expanduser(), "test")
    for index, label in enumerate(CLASSES):
        matches = [path for path, value in zip(paths, labels) if value == index]
        source = matches[args.index]
        image = Image.open(source).convert("RGB")
        small = cv2.resize(np.asarray(image), (450, 600), interpolation=cv2.INTER_AREA)
        target = output / f"{label.lower()}.png"
        Image.fromarray(small).save(target, optimize=True)
        print(f"{target} {target.stat().st_size / 1024:.0f} KB from {source}")


if __name__ == "__main__":
    main()
