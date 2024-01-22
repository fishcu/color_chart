import cv2
import numpy as np
from cv2 import mcc
import os
import matplotlib.pyplot as plt
from skimage import color

from chart import *
from loss import *

# def get_test_chart_rgb(image):
#     height, width = image.shape[:2]

#     # Create a CCheckerDetector object
#     detector = mcc.CCheckerDetector.create()

#     # Detect the color chart in the image
#     if not detector.process(image, mcc.MCC24):
#         raise RuntimeError("Detection failed")

#     # Get best detected chart
#     cc = detector.getBestColorChecker()

#     # Get RGB values
#     rgb_raw = cc.getChartsRGB()

#     rgb = np.zeros((24, 3))
#     for i in range(24):
#         rgb[i, 0] = rgb_raw[3 * i, 1]
#         rgb[i, 1] = rgb_raw[3 * i + 1, 1]
#         rgb[i, 2] = rgb_raw[3 * i + 2, 1]

#     # Return box around CC so that cropped image can be exported easily
#     boxes = cc.getBox()
#     min_x = max(0, min(b[0] for b in boxes))
#     max_x = min(width, max(b[0] for b in boxes))
#     min_y = max(0, min(b[1] for b in boxes))
#     max_y = min(height, max(b[1] for b in boxes))

#     return rgb, (min_x, max_x, min_y, max_y)


# def save_annotated_image(image, rgb_values, crop_box, image_path):
#     min_x, max_x, min_y, max_y = crop_box
#     cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]

#     height, width = cropped_image.shape[:2]
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.4
#     font_thickness = 1

#     for i in range(4):
#         for j in range(6):
#             index = i * 6 + j
#             r, g, b = rgb_values[index]

#             # Calculate evenly spaced coordinates
#             x_position = int((j + 0.5) * width / 6)
#             y_position = int((i + 0.5) * height / 4)

#             # Calculate text size for centering
#             text_size = cv2.getTextSize(
#                 "R: 255", font, font_scale, font_thickness)[0]

#             # Calculate centered position
#             x_position -= text_size[0] // 2
#             y_position += text_size[1] // 2

#             position = (x_position, y_position)

#             # Calculate luminance (average of RGB components)
#             luminance = np.mean(rgb_values[index])

#             # Determine text color based on luminance threshold
#             text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

#             r = int(round(r))
#             g = int(round(g))
#             b = int(round(b))

#             # Draw R, G, B values on separate lines
#             cv2.putText(cropped_image, f"R: {r}", (position[0], position[1] - 2 * text_size[1]), font, font_scale,
#                         text_color, font_thickness, cv2.LINE_AA)
#             cv2.putText(cropped_image, f"G: {g}", position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
#             cv2.putText(cropped_image, f"B: {b}", (position[0], position[1] + 2 * text_size[1]), font, font_scale,
#                         text_color, font_thickness, cv2.LINE_AA)

#     base_name, ext = os.path.splitext(image_path)
#     annotated_path = f"{base_name}_annotated{ext}"
#     cv2.imwrite(annotated_path, cropped_image)
#     print(f"Annotated image saved at: {annotated_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <ref_image_path> <test_image_path>")
        sys.exit(1)

    ref_image_path = sys.argv[1]
    test_image_path = sys.argv[2]

    ref_image = cv2.imread(ref_image_path)
    test_image = cv2.imread(test_image_path)

    # Process the image and get the RGB matrix
    # ref_rgb_matrix, ref_crop_box = get_test_chart_rgb(ref_image)
    # test_rgb_matrix, test_crop_box = get_test_chart_rgb(test_image)

    # Save annotated image
    # save_annotated_image(ref_image, ref_rgb_matrix,
    #                      ref_crop_box, ref_image_path)
    # save_annotated_image(test_image, test_rgb_matrix,
    #                      test_crop_box, test_image_path)

    # Flatten values and normalize
    # ref_rgb_values = ref_rgb_matrix.reshape((-1, 3)) / 255.0
    # test_rgb_values = test_rgb_matrix.reshape((-1, 3)) / 255.0

    ref_rgb_values = get_chart_rgb(ref_image) / 255.0
    test_rgb_values = get_chart_rgb(test_image) / 255.0

    # Convert to LAB colors
    ref_lab_values = color.rgb2lab(ref_rgb_values)
    test_lab_values = color.rgb2lab(test_rgb_values)

    # Print loss
    loss = loss_no_correction(ref_lab_values, test_lab_values)
    print(f"Total color discrepancy: {round(loss, 2)}")

    # Extracting L*, a*, and b* values
    ref_L_values = ref_lab_values[:, 0]
    ref_a_values = ref_lab_values[:, 1]
    ref_b_values = ref_lab_values[:, 2]
    test_L_values = test_lab_values[:, 0]
    test_a_values = test_lab_values[:, 1]
    test_b_values = test_lab_values[:, 2]

    # Plotting the first graph with a* and b*
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(ref_a_values, ref_b_values, c=ref_rgb_values, edgecolors='black',
                marker='o', s=200, label=os.path.basename(ref_image_path))
    plt.scatter(test_a_values, test_b_values, c=test_rgb_values, edgecolors='black',
                marker='s', s=200, label=os.path.basename(test_image_path))
    plt.xlabel('red-green')
    plt.ylabel('yellow-blue')
    plt.title('a* vs b*')
    plt.grid(True)
    plt.legend()

    # Connect corresponding samples with thin black lines
    for i in range(len(ref_a_values)):
        plt.plot([ref_a_values[i], test_a_values[i]], [
                 ref_b_values[i], test_b_values[i]], 'k-', linewidth=0.5)

    # Plotting the second graph with L* and chroma
    plt.subplot(1, 2, 2)
    plt.scatter(np.sqrt(ref_a_values**2 + ref_b_values**2), ref_L_values, c=ref_rgb_values,
                edgecolors='black', marker='o', s=200, label=os.path.basename(ref_image_path))
    plt.scatter(np.sqrt(test_a_values**2 + test_b_values**2), test_L_values, c=test_rgb_values,
                edgecolors='black', marker='s', s=200, label=os.path.basename(test_image_path))
    plt.xlabel('chroma')
    plt.ylabel('luminosity')
    plt.title('chroma vs L*')
    plt.grid(True)
    plt.legend()

    # Connect corresponding samples with thin black lines
    for i in range(len(ref_a_values)):
        plt.plot([np.sqrt(ref_a_values[i]**2 + ref_b_values[i]**2), np.sqrt(test_a_values[i]**2 + test_b_values[i]**2)],
                 [ref_L_values[i], test_L_values[i]], 'k-', linewidth=0.5)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()
