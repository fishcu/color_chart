import cv2
import numpy as np
from cv2 import mcc
import os

def save_rgb_matrix(image_path, rgb_matrix):
    base_name, ext = os.path.splitext(image_path)
    txt_path = f"{base_name}_rgb_matrix.txt"
    with open(txt_path, 'w') as file:
        for row in rgb_matrix:
            file.write(" ".join(map(str, row)) + "\n")
    print(f"RGB matrix saved at: {txt_path}")

def save_annotated_image(image, rgb_values, crop_box, image_path):
    min_x, max_x, min_y, max_y = crop_box
    cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]

    height, width = cropped_image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2

    for i in range(4):
        for j in range(6):
            index = i * 6 + j
            r, g, b = rgb_values[index]

            # Calculate evenly spaced coordinates
            x_position = int((j + 0.5) * width / 6)
            y_position = int((i + 0.5) * height / 4)

            # Calculate text size for centering
            text_size = cv2.getTextSize("R: 255", font, font_scale, font_thickness)[0]

            # Calculate centered position
            x_position -= text_size[0] // 2
            y_position += text_size[1] // 2

            position = (x_position, y_position)

            # Calculate luminance (average of RGB components)
            luminance = np.mean(rgb_values[index])

            # Determine text color based on luminance threshold
            text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

            # Draw R, G, B values on separate lines
            cv2.putText(cropped_image, f"R: {r}", (position[0], position[1] - 2 * text_size[1]), font, font_scale,
                        text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(cropped_image, f"G: {g}", position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(cropped_image, f"B: {b}", (position[0], position[1] + 2 * text_size[1]), font, font_scale,
                        text_color, font_thickness, cv2.LINE_AA)

    base_name, ext = os.path.splitext(image_path)
    annotated_path = f"{base_name}_annotated{ext}"
    cv2.imwrite(annotated_path, cropped_image)
    print(f"Annotated image saved at: {annotated_path}")

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

    rgb = np.zeros((24, 3), dtype=int)
    for i in range(24):
        rgb[i, 0] = int(round(rgb_raw[3 * i, 1]))
        rgb[i, 1] = int(round(rgb_raw[3 * i + 1, 1]))
        rgb[i, 2] = int(round(rgb_raw[3 * i + 2, 1]))

    # Crop image to around where color chart was detected
    boxes = cc.getBox()
    min_x = max(0, min(b[0] for b in boxes))
    max_x = min(width, max(b[0] for b in boxes))
    min_y = max(0, min(b[1] for b in boxes))
    max_y = min(height, max(b[1] for b in boxes))

    return rgb, (min_x, max_x, min_y, max_y)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    original_image = cv2.imread(image_path)

    # Process the image and get the RGB matrix
    rgb_matrix, crop_box  = process_image(original_image)

    # Save the RGB matrix as a text file
    save_rgb_matrix(image_path, rgb_matrix)

    # Save annotated image
    save_annotated_image(original_image, rgb_matrix, crop_box, image_path)

    # loaded_rgb_matrix = load_rgb_matrix(f"{os.path.splitext(image_path)[0]}_rgb_matrix.txt")

    # if loaded_rgb_matrix is not None:
    #     # Now you can use loaded_rgb_matrix as needed
    #     print("Loaded RGB Matrix:")
    #     print(loaded_rgb_matrix)
