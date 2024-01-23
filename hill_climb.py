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

from skimage import io, color

from chart import *
from loss import *
from kodak_extract import *


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


class CCChartHandler(FileSystemEventHandler):
    def __init__(self, ref_lab):
        super().__init__()
        self.min_loss = float('inf')  # Initialize with positive infinity
        self.ref_lab = ref_lab

    def on_created(self, event):
        if event.is_directory:
            return  # Ignore directory creation events

        if not is_jpeg_file(event.src_path):
            return  # Ignore non-jpeg files

        time.sleep(0.2)

        test_file_path = str(Path(event.src_path)).replace('\\', '/')
        test_img = cv2.imread(test_file_path)

        test_lab = get_chart_lab(test_img)

        current_loss = loss_no_correction(ref_lab, test_lab)

        extra_note = ""
        if current_loss + 1.e-3 < self.min_loss:
            self.min_loss = current_loss
            extra_note = " (NEW BEST)"

        print(f"New file added: {test_file_path}")
        print(f"Loss of the newest added file: {round(current_loss, 2)}"
              + extra_note)
        print(f"Lowest loss so far: {round(self.min_loss, 2)}")

        # Move the file to a different directory with appended loss
        new_file_path_with_loss = move_file_with_loss(
            test_file_path, current_loss)
        print(f"File moved to: {new_file_path_with_loss}")
        print("=" * 30)


class KodakChartHandler(FileSystemEventHandler):
    def __init__(self, ref_lab, test_points):
        super().__init__()
        self.min_loss = float('inf')  # Initialize with positive infinity
        self.ref_lab = ref_lab
        self.test_processor = KodakExtractor(test_points)

    def on_created(self, event):
        if event.is_directory:
            return  # Ignore directory creation events

        if not is_jpeg_file(event.src_path):
            return  # Ignore non-jpeg files

        time.sleep(0.2)

        test_file_path = str(Path(event.src_path)).replace('\\', '/')
        test_img = cv2.imread(test_file_path)

        test_lab = self.test_processor.get_lab_values(test_img)

        current_loss = loss_no_correction(ref_lab, test_lab)

        extra_note = ""
        if current_loss + 1.e-3 < self.min_loss:
            self.min_loss = current_loss
            extra_note = " (NEW BEST)"

        print(f"New file added: {test_file_path}")
        print(f"Loss of the newest added file: {round(current_loss, 2)}"
              + extra_note)
        print(f"Lowest loss so far: {round(self.min_loss, 2)}")

        # Move the file to a different directory with appended loss
        new_file_path_with_loss = move_file_with_loss(
            test_file_path, current_loss)
        print(f"File moved to: {new_file_path_with_loss}")
        print("=" * 30)


def watch_directory(event_handler, directory_path):
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
    if len(sys.argv) != 3 and len(sys.argv) != 11:
        print("Usage: python script.py <path to ref image> <directory_path>")
        print("For Kodak: python script.py <ref_image_path> p1 p2 p3 p4 <directory_path> p1 p2 p3 p4")
        sys.exit(1)

    kodak_mode = len(sys.argv) == 11

    if not kodak_mode:
        directory_path = sys.argv[2]
    else:
        directory_path = sys.argv[6]
    if not os.path.isdir(directory_path):
        print("Error: The specified path is not a directory.")
        sys.exit(1)

    ref_path = sys.argv[1]
    ref_img = cv2.imread(ref_path)
    if not kodak_mode:
        ref_lab = get_chart_lab(ref_img)
    else:
        ref_points = [tuple(map(int, point.split(',')))
                      for point in sys.argv[2:6]]
        ref_extractor = KodakExtractor(ref_points)
        ref_lab = ref_extractor.get_lab_values(ref_img)

    print(f"Reference image analyzed successfully. Waiting for new images in {
          directory_path}")

    if not kodak_mode:
        event_handler = CCChartHandler(ref_lab)
    else:
        event_handler = KodakChartHandler(ref_lab, [tuple(map(int, point.split(',')))
                                                    for point in sys.argv[7:11]])
    watch_directory(event_handler, directory_path)
