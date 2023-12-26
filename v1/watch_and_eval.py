import os
import sys
import time
from pathlib import Path
import time

import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import numpy as np

# from PIL import Image
# from PIL.ExifTags import TAGS

from skimage import io, color

from cc_extract_rgb import process_image
from get_loss import compute_delta_e_values

# Function to move the file to a different directory with appended loss


def srgb2lab(data):
    return color.rgb2lab(data / 255.0)


def lab2srgb(data):
    return color.lab2rgb(data) * 255.0


def calculate_rmse(img1, img2):
    """
    Calculate Root Mean Squared Error (RMSE) between two images.

    Parameters:
    - img1: NumPy array representing the first image.
    - img2: NumPy array representing the second image.

    Returns:
    - rmse: Root Mean Squared Error between the two images.
    """
    # Ensure the images have the same shape
    assert img1.shape == img2.shape, "Images must have the same shape"

    # Calculate squared difference between pixel values
    squared_diff = (img1.astype(np.float64) - img2.astype(np.float64)) ** 2

    # Calculate mean squared error
    mean_squared_error = np.mean(squared_diff)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error)

    return rmse


def move_file_with_loss(file_path, loss):
    base_dir = os.path.dirname(file_path)
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))

    # Create a new directory for moved files if it doesn't exist
    moved_dir = os.path.join(base_dir, 'processed_files')
    os.makedirs(moved_dir, exist_ok=True)

    # Construct the new file name with appended loss
    new_file_name = f"{file_name}_loss{loss:.3f}{file_ext}"
    new_file_path = os.path.join(moved_dir, new_file_name)

    # Move the file to the new directory
    try:
        os.rename(file_path, new_file_path)
    except FileExistsError:
        # Ignore the error silently
        pass
    return new_file_path


def is_jpeg_file(file_path):
    # Get the file extension (lowercased for case-insensitive comparison)
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file extension indicates a JPEG file
    return file_extension in {'.jpg', '.jpeg', '.jpe', '.jfif'}

# Custom event handler for file system events


class MyHandler(FileSystemEventHandler):
    def __init__(self, ref_rgb):
        super().__init__()
        self.min_loss = float('inf')  # Initialize with positive infinity
        self.ref_rgb = ref_rgb

    def on_created(self, event):
        if event.is_directory:
            return  # Ignore directory creation events

        if not is_jpeg_file(event.src_path):
            return  # Ignore non-jpeg files

        time.sleep(1)

        test_file_path = str(Path(event.src_path)).replace('\\', '/')
        test_image = cv2.imread(test_file_path)

        test_rgb, _ = process_image(test_image)

        # print(test_rgb)

        current_loss = compute_delta_e_values(ref_rgb, test_rgb)[-1]

        if current_loss + 1.e-3 < self.min_loss:
            self.min_loss = current_loss
            # read_exif(test_file_path)

        print(f"New file added: {test_file_path}")
        print(f"Loss of the newest added file: {round(current_loss, 2)}"
              + (" (NEW BEST)" if self.min_loss == current_loss else ""))
        print(f"Lowest loss so far: {round(self.min_loss, 2)}")

        # Move the file to a different directory with appended loss
        new_file_path_with_loss = move_file_with_loss(
            test_file_path, current_loss)
        print(f"File moved to: {new_file_path_with_loss}")
        print("=" * 30)


class RMSEEval(FileSystemEventHandler):
    def __init__(self, ref_img):
        super().__init__()
        self.min_loss = float('inf')  # Initialize with positive infinity
        self.ref_img = ref_img

    def on_created(self, event):
        if event.is_directory:
            return  # Ignore directory creation events

        if not is_jpeg_file(event.src_path):
            return  # Ignore non-jpeg files

        time.sleep(0.5)

        test_file_path = str(Path(event.src_path)).replace('\\', '/')
        test_image = io.imread(test_file_path)

        test_image = srgb2lab(test_image)

        current_loss = calculate_rmse(self.ref_img, test_image)

        if current_loss + 1.e-3 < self.min_loss:
            self.min_loss = current_loss
            # read_exif(test_file_path)

        print(f"New file added: {test_file_path}")
        print(f"Loss of the newest added file: {round(current_loss, 2)}"
              + (" (NEW BEST)" if self.min_loss == current_loss else ""))
        print(f"Lowest loss so far: {round(self.min_loss, 2)}")

        # Move the file to a different directory with appended loss
        new_file_path_with_loss = move_file_with_loss(
            test_file_path, current_loss)
        print(f"File moved to: {new_file_path_with_loss}")
        print("=" * 30)



def watch_directory(ref_img, directory_path):
    event_handler = RMSEEval(ref_img)
    observer = Observer()
    observer.schedule(event_handler, path=directory_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

# def pretty_print_exif(exif_data):
#     for tag, value in exif_data.items():
#         tag_name = TAGS.get(tag, tag)
#         print(f"{tag_name}: {value}")

# def read_exif(image_path):
#     try:
#         with Image.open(image_path) as img:
#             exif_data = img._getexif()
#             if exif_data is not None:
#                 pretty_print_exif(exif_data)
#             else:
#                 print("No EXIF data found.")
#     except Exception as e:
#         print(f"Error reading EXIF data: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <reference_image> <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[2]
    if not os.path.isdir(directory_path):
        print("Error: The specified path is not a directory.")
        sys.exit(1)

    ref_path = sys.argv[1]
    ref_img = io.imread(ref_path)
    # ref_rgb, _ = process_image(ref_img)

    ref_img = srgb2lab(ref_img)

    print(f"Reference image analyzed successfully. Waiting for new images in {directory_path}")

    watch_directory(ref_img, directory_path)
