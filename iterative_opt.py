import os

import cv2
import numpy as np

from natsort import natsorted

from chart import *
from loss import *
from kodak_extract import *

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 and len(sys.argv) != 12:
        print("Usage: python script.py <path to reference image> <path to test image> <image directory>")
        print("For Kodak: python script.py <ref_image_path> p1 p2 p3 p4 <test_image_path> p1 p2 p3 p4 <img_dir>")
        sys.exit(1)

    kodak_mode = len(sys.argv) == 12

    # Y in previous definition
    print("Processing reference image...")
    ref_img_path = sys.argv[1]
    ref_img = cv2.imread(ref_img_path)
    if not kodak_mode:
        ref_lab = get_chart_lab(ref_img)
    else:
        ref_points = [tuple(map(int, point.split(',')))
                      for point in sys.argv[2:6]]
        ref_extractor = KodakExtractor(ref_points)
        ref_lab = ref_extractor.get_lab_values(ref_img)

    num_color_pads = len(ref_lab)

    # X in previous definition
    print("Processing test image...")
    if not kodak_mode:
        test_img_path = sys.argv[2]
        test_img = cv2.imread(test_img_path)
        test_lab = get_chart_lab(test_img)
    else:
        test_img_path = sys.argv[6]
        test_img = cv2.imread(test_img_path)
        test_points = [tuple(map(int, point.split(',')))
                      for point in sys.argv[7:11]]
        test_extractor = KodakExtractor(test_points)
        test_lab = test_extractor.get_lab_values(test_img)

    assert len(test_lab) == num_color_pads

    # Get "D" vectors
    if not kodak_mode:
        d_img_dir = sys.argv[3]
    else:
        d_img_dir = sys.argv[11]
    files = os.listdir(d_img_dir)
    img_files = [file for file in files if file.lower().endswith(
        ('.jpg', '.jpeg', '.png'))]
    img_files = natsorted(img_files)
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
        if not kodak_mode:
            deltas[:, i, :] = get_chart_lab(img) - test_lab
        else:
            deltas[:, i, :] = test_extractor.get_lab_values(img) - test_lab

    # Solve
    lambda_s = 1.0
    lhs = lambda_s * np.identity(num_controls)
    rhs = np.zeros((num_controls,))
    for d in range(3):
        lhs += np.matmul(np.transpose(deltas[:, :, d]), deltas[:, :, d])
        rhs += np.dot(np.transpose(deltas[:, :, d]),
                      ref_lab[:, d] - test_lab[:, d])
    w = np.linalg.solve(lhs, rhs)
    print(f"Color discrepancy before adjustments:    {
          round(loss(test_lab, ref_lab, deltas, np.zeros((num_controls))), 2)}")
    print(f"Idealized discrepancy after adjustments: {
          round(loss(test_lab, ref_lab, deltas, w), 2)}")

    # Show result
    max_length = max(len(file) for file in img_files)
    max_adj = 2
    w = w.round()
    if np.any(w != 0):
        print("=========================================\nDo the following adjustments and iterate:")
        for i in range(num_controls):
            if abs(w[i]) > max_adj:
                w[i] = max_adj * np.sign(w[i])
            if w[i] != 0:
                print(f"{img_files[i]:<{max_length + 1}}: Adjust by {w[i]}")
    else:
        print("You found approximately optimal settings! It's still recommended to try a few settings around the current optimum to see if it can be improved a bit.")
