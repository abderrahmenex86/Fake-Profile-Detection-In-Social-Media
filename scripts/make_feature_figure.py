import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="dataset/train/REAL/0.png")
    parser.add_argument("--output", default="results/plots/fig_features_example.png")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        candidates = sorted(Path("dataset/train").rglob("*.png"))
        path = candidates[0]

    image = imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    small = cv2.resize(image, (450, 600), interpolation=cv2.INTER_AREA)
    gray = rgb2gray(small)
    _, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
    )

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    keypoints = cv2.SIFT_create().detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)[0]
    sift_image = cv2.drawKeypoints(
        bgr,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    sift_image = cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB)

    raw = cv2.resize(image, (150, 200), interpolation=cv2.INTER_AREA)

    figure, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original screenshot\n600×800")
    axes[1].imshow(hog_image, cmap="magma")
    axes[1].set_title("HOG\n34,992 dims")
    axes[2].imshow(sift_image)
    axes[2].set_title(f"SIFT keypoints\n32,768 dims ({len(keypoints)} keypoints)")
    axes[3].imshow(raw)
    axes[3].set_title("Raw pixels (resized)\n90,000 dims")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
