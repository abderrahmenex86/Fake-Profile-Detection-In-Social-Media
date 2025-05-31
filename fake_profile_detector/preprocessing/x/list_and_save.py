import os

from datasets import load_dataset
from tqdm import tqdm

from ...configs.general import BASE_DIR


def download_x_dataset():
    dataset_dir = os.path.join(BASE_DIR, "x")
    os.makedirs(dataset_dir, exist_ok=True)

    dataset = load_dataset("drveronika/x_fake_profile_detection")

    total_files = 0
    idx = 0
    for split_name, split_data in dataset.items():
        for sample in tqdm(split_data, desc=f"Saving {split_name}"):
            image = sample.get("image")
            label = sample.get("label")

            if image and label is not None:
                filename = f"{idx:05d}_{label}.png"
                filepath = os.path.join(dataset_dir, filename)
                image.save(filepath)
                total_files += 1
                idx += 1
                # tqdm.write(f"Saved {filepath}")

    print(f"Saved {total_files} images to {dataset_dir}")


if __name__ == "__main__":
    download_x_dataset()
