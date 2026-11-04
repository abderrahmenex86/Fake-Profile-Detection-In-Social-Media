from PIL import Image

from fpd.data import CLASSES, limit_split, load_split, search_indices


def test_load_split(tmp_path):
    for label in CLASSES:
        folder = tmp_path / "train" / label
        folder.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (10, 10), (index, 0, 0)).save(folder / f"{index}.png")

    paths, labels = load_split(tmp_path, "train")
    assert len(paths) == 8
    assert len(labels) == 8

    limited_paths, limited_labels = limit_split(paths, labels, 4, 0)
    assert len(limited_paths) == 4
    assert len(limited_labels) == 4

    indices = search_indices(labels, 4, 0)
    assert len(indices) == 4
