import glob
import os

from PIL import Image
from tqdm import tqdm


def resize_images_in_directory(input_dir, output_dir, target_size=(600, 800)):
    """
    Resize all images in a directory to target size.
    
    Parameters:
        input_dir: Source directory containing images
        output_dir: Destination directory for resized images
        target_size: Target size as (width, height)
    
    Returns:
        Number of successfully resized images
    """
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_files:
        print(f"No images found in {input_dir}")
        return 0

    print(f"Resizing {len(image_files)} images from {input_dir} to {target_size}")

    successful = 0
    failed = 0

    for image_path in tqdm(
        image_files, desc=f"Processing {os.path.basename(input_dir)}"
    ):
        try:
            with Image.open(image_path) as img:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)

                successful += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            failed += 1

    print(f"✓ Successfully resized: {successful} images")
    if failed > 0:
        print(f"✗ Failed to resize: {failed} images")
    
    return successful


def resize_test_dataset(
    base_dir="/home/abderrahmene/datasets/small",
    output_dir="/home/abderrahmene/datasets/small_resized",
    target_size=(600, 800),
):
    """
    Resize the entire test dataset (bot, cyborg, real, verified subdirectories).
    
    Parameters:
        base_dir: Original dataset directory
        output_dir: Output directory for resized dataset
        target_size: Target size as (width, height)
    """
    print(f"Resizing dataset from {base_dir} to {output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")

    # Define the 4 classes
    classes = ["bot", "cyborg", "real", "verified"]

    # Check if input directories exist
    missing_dirs = []
    for class_name in classes:
        class_input = os.path.join(base_dir, class_name)
        if not os.path.exists(class_input):
            missing_dirs.append(class_input)

    if missing_dirs:
        raise ValueError(f"Missing directories: {missing_dirs}")

    # Resize images in each class directory
    total_resized = 0
    for class_name in classes:
        class_input = os.path.join(base_dir, class_name)
        class_output = os.path.join(output_dir, class_name)
        
        resized_count = resize_images_in_directory(class_input, class_output, target_size)
        total_resized += resized_count

    print(f"\nDataset resizing completed!")
    print(f"Total images resized: {total_resized}")
    print(f"Resized dataset saved to: {output_dir}")


def verify_resizing(output_dir="/home/abderrahmene/datasets/small_resized"):
    """
    Verify that the resizing worked correctly for all 4 classes.
    """
    print("\nVerifying resized images...")

    classes = ["bot", "cyborg", "real", "verified"]

    for class_name in classes:
        resized_dir = os.path.join(output_dir, class_name)
        if os.path.exists(resized_dir):
            image_files = glob.glob(os.path.join(resized_dir, "*"))
            if image_files:
                # Check first image
                sample_image = image_files[0]
                with Image.open(sample_image) as img:
                    print(
                        f"✓ {class_name} ({len(image_files)} images) - sample ({os.path.basename(sample_image)}): {img.size}"
                    )
            else:
                print(f"✗ No images found in {resized_dir}")
        else:
            print(f"✗ Directory not found: {resized_dir}")


def main():
    """
    Main function to resize the test dataset.
    """
    try:
        resize_test_dataset()
        verify_resizing()
    except Exception as e:
        print(f"Error resizing dataset: {e}")
        raise


if __name__ == "__main__":
    main()
