import os

import cv2
import numpy as np

from cv2 import mcc
from skimage import color


def get_chart_rgb(image):
    height, width = image.shape[:2]

    # Create a CCheckerDetector object
    detector = mcc.CCheckerDetector.create()

    # Detect the color chart in the image
    if not detector.process(image, mcc.MCC24):
        raise RuntimeError("Detection failed")

    # Get best detected chart
    cc = detector.getBestColorChecker()

    # Get RGB values
    rgb_raw = cc.getChartsRGB()

    rgb = np.zeros((24, 3))
    for i in range(24):
        rgb[i, 0] = rgb_raw[3 * i, 1]
        rgb[i, 1] = rgb_raw[3 * i + 1, 1]
        rgb[i, 2] = rgb_raw[3 * i + 2, 1]

    return rgb.reshape((-1, 3))


def get_chart_lab(image):
    rgb = get_chart_rgb(image)
    return color.rgb2lab(rgb / 255.0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python script.py <path to reference image> <path to test image> <image directory>")
        sys.exit(1)

    ref_img_path = sys.argv[1]
    ref_img = cv2.imread(ref_img_path)
    # Y in previous definition
    print("Processing reference image...")
    ref_lab = get_chart_lab(ref_img)

    num_color_pads = len(ref_lab)

    test_img_path = sys.argv[2]
    test_img = cv2.imread(test_img_path)
    # X in previous definition
    print("Processing test image...")
    test_lab = get_chart_lab(test_img)

    assert len(test_lab) == num_color_pads

    # Get "D" vectors
    d_img_dir = sys.argv[3]
    files = os.listdir(d_img_dir)
    img_files = [file for file in files if file.lower().endswith(
        ('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print(f"No JPEG files found in '{d_img_dir}'.")
        sys.exit(1)

    num_controls = len(img_files)
    print(f"Found {num_controls} delta images.")
    deltas = np.zeros((num_color_pads, num_controls, 3))
    for i, file in enumerate(img_files):
        print(f"\tProcessing {file}...")
        file_path = os.path.join(d_img_dir, file)
        img = cv2.imread(file_path)
        deltas[:, i, :] = get_chart_lab(img) - test_lab

    # Solve
    lambda_s = 1.0
    lhs = lambda_s * np.identity(num_controls)
    rhs = np.zeros((num_controls,))
    for d in range(3):
        lhs += np.matmul(np.transpose(deltas[:, :, d]), deltas[:, :, d])
        rhs += np.dot(np.transpose(deltas[:, :, d]),
                      ref_lab[:, d] - test_lab[:, d])
    w = np.linalg.solve(lhs, rhs)

    # Show result
    print("=========================================\nDo the following adjustments and iterate:")
    for i in range(num_controls):
        print(f"{img_files[i]:<40}: Adjust by {round(w[i])}")
