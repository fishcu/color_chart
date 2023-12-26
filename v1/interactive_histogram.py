import os
import sys
import time
from pathlib import Path
import time
import shutil

import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

from ruderman import rgb_to_lab


def move_file_with_loss(file_path, loss):
    base_dir = os.path.dirname(file_path)
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))

    # Create a new directory for moved files if it doesn't exist
    moved_dir = os.path.join(base_dir, 'processed_files')
    os.makedirs(moved_dir, exist_ok=True)

    # Construct the new file name with appended loss
    new_file_name = f"{file_name}_loss{loss:.3f}{file_ext}"
    new_file_path = os.path.join(moved_dir, new_file_name)

    # Move and overwrite
    try:
        shutil.move(file_path, new_file_path)
    except shutil.Error:
        # Handle any error that shutil.move might raise
        try:
            # If an error occurs, try to copy and overwrite the file
            shutil.copy2(file_path, new_file_path)
            os.remove(file_path)  # Remove the original file after copying
        except Exception as e:
            print(f"Error: {e}")
    return new_file_path


def is_jpeg_file(file_path):
    # Get the file extension (lowercased for case-insensitive comparison)
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file extension indicates a JPEG file
    return file_extension in {'.jpg', '.jpeg', '.jpe', '.jfif'}

# Custom event handler for file system events


class MyHandler(FileSystemEventHandler):
    def __init__(self, ref_hist, dynamic_range):
        super().__init__()
        self.min_loss = float('inf')  # Initialize with positive infinity
        self.ref_hist = ref_hist
        self.dynamic_range = dynamic_range

    def on_created(self, event):
        if event.is_directory:
            return  # Ignore directory creation events

        if not is_jpeg_file(event.src_path):
            return  # Ignore non-jpeg files

        time.sleep(0.2)

        test_file_path = str(Path(event.src_path)).replace('\\', '/')
        test_image = cv2.imread(test_file_path)

        test_lab = rgb_to_lab(test_image)
        # test_lab = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)

        losses = []
        for c in range(3):
            h = cv2.calcHist(
                [test_lab], [c], None, [256], self.dynamic_range[c])
            cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            losses.append(cv2.compareHist(ref_hist[c], h, cv2.HISTCMP_BHATTACHARYYA))
            # losses.append(wasserstein_distance(ref_hist[c], h.flatten()))
        print(losses)

        # RMS loss
        current_loss = 1000 * np.sqrt(np.mean(np.array(losses)**2))

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

# Main function to watch the directory


def watch_directory(directory_path, ref_hist, dynamic_range):
    event_handler = MyHandler(ref_hist, dynamic_range)
    observer = Observer()
    observer.schedule(event_handler, path=directory_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <reference_image> <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[2]
    if not os.path.isdir(directory_path):
        print("Error: The specified path is not a directory.")
        sys.exit(1)

    ref_path = sys.argv[1]
    ref_img = cv2.imread(ref_path)
    ref_lab = rgb_to_lab(ref_img)
    # ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)

    ref_hist = []
    dynamic_range = []
    for c in range(3):
        dynamic_range.append(
            [np.min(ref_lab[:, :, c]), np.max(ref_lab[:, :, c])])
        print(dynamic_range[c])
        h = cv2.calcHist(
            [ref_lab], [c], None, [256], dynamic_range[c])
        cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        ref_hist.append(h.flatten())

    # colors = ['blue', 'green', 'red']
    # for i, hist in enumerate(ref_hist):
    #     plt.plot(hist, color=colors[i])

    # plt.title('Histograms for Each Channel')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.legend(['L', 'a', 'b'])
    # plt.show()

    print(f"Reference image analyzed successfully. Waiting for new images in {
          directory_path}")

    watch_directory(directory_path, ref_hist, dynamic_range)
