from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread


def extract_hog_features(
    image_path, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)
):
    try:
        img = imread(image_path)

        if img.ndim == 3 and img.shape[2] == 4:
            from skimage.color import rgba2rgb

            img = rgba2rgb(img)

        img_gray = rgb2gray(img)

        features = hog(
            img_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True,
            channel_axis=None,
        )
        return features
    except Exception as _:
        return None
