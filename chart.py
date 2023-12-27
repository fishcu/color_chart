import numpy as np

from cv2 import mcc
from skimage import color


def get_chart_rgb(image):
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
