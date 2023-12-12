import numpy as np
from tps import ThinPlateSpline
import sys
from skimage import io, color
from cv2 import mcc
import cv2

from cc_extract_rgb import save_annotated_image

def process_image(image):
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

    # Crop image to around where color chart was detected
    boxes = cc.getBox()
    min_x = max(0, min(b[0] for b in boxes))
    max_x = min(width, max(b[0] for b in boxes))
    min_y = max(0, min(b[1] for b in boxes))
    max_y = min(height, max(b[1] for b in boxes))

    return rgb, (min_x, max_x, min_y, max_y)


def create_tps(s, t):
    tps = ThinPlateSpline(alpha=0.0)
    tps.fit(s, t)
    return tps


def srgb2lab(data):
    return color.rgb2lab(data / 255.0)


def lab2srgb(data):
    return color.lab2rgb(data) * 255.0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py source_img.jpg target_img.jpg input_image.jpg")
        sys.exit(1)

    source_path, target_path, image_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Load source and target RGB matrices from argv
    source, _ = process_image(cv2.imread(source_path))
    target, target_crop = process_image(cv2.imread(target_path))

    # save_annotated_image(cv2.imread(target_path), target, target_crop, "target_annotated.jpg")

    if source is None or target is None:
        sys.exit(1)

    print("target in rgb")
    print(target)

    source = srgb2lab(source)
    target = srgb2lab(target)

    print("source in cielab")
    print(source)
    print("target in cielab")
    print(target)

    tps = create_tps(source, target)

    print("mapped source in lab")
    print(tps.transform(source))

    print("mapped source in rgb")
    print(lab2srgb(tps.transform(source)))

    # Load image from argv
    input_image = io.imread(image_path)
    # print("min max:")
    # print(np.min(input_image), np.max(input_image))

    # Convert image to other color space
    image_conv = srgb2lab(input_image)

    print(image_conv.shape)

    # Convert image to linear array of points
    linear_array = image_conv.reshape((-1, 3))

    print(linear_array.shape)

    # Transform image
    image_out_linear = tps.transform(linear_array)
    # print("min max:")
    # print(np.min(image_out_linear), np.max(image_out_linear))

    # Shape transformed image back to original dimensions
    transformed_image_conv = image_out_linear.reshape(image_conv.shape)

    # Convert transformed image from CIELAB to RGB
    transformed_image_rgb = lab2srgb(transformed_image_conv)

    print(transformed_image_rgb.shape)

    # transformed_image_rgb = lab2srgb(image_conv)

    # Display the transformed image
    io.imshow(transformed_image_rgb / 255.0)
    io.show()

    # Save transformed image to disk
    io.imsave("transformed_image.jpg", transformed_image_rgb.astype(np.uint8))
