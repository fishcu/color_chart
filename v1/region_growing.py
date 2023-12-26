import sys
import cv2
import numpy as np
from skimage.color import deltaE_ciede2000

def display_usage():
    print("Usage: python script.py <image_path>")
    sys.exit(1)


def chebychev(point1, point2):
    return max(abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))


def shrink_mask(mask, shrink_pixels):
    kernel = np.ones((2 * shrink_pixels + 1, 2 *
                     shrink_pixels + 1), dtype=np.uint8)
    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    return shrunk_mask


def region_growing(image, seed, threshold, min_size, max_distance):
    height, width, _ = image.shape
    visited = np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    stack = [seed]
    pixel_count = 0

    while stack:
        current_point = stack.pop()
        if not (0 <= current_point[0] < height and 0 <= current_point[1] < width):
            continue

        if visited[current_point[0], current_point[1]] == 1:
            continue

        current_color = image[current_point[0], current_point[1]]
        color_distance = deltaE_ciede2000(current_color, image[seed])
        geom_dist = chebychev(current_point, seed)

        if color_distance <= threshold and geom_dist <= max_distance:
            mask[current_point[0], current_point[1]] = 255
            visited[current_point[0], current_point[1]] = 1
            pixel_count += 1

            # Add neighboring pixels to the stack
            stack.append((current_point[0] - 1, current_point[1]))
            stack.append((current_point[0] + 1, current_point[1]))
            stack.append((current_point[0], current_point[1] - 1))
            stack.append((current_point[0], current_point[1] + 1))

    # Return an empty mask if pixel count is below the threshold
    if pixel_count < min_size:
        return np.zeros_like(mask)
    else:
        return mask


def main():
    if len(sys.argv) != 2:
        display_usage()

    image_path = sys.argv[1]

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image.")
        sys.exit(1)

    # Resize the image to 1/8 of its original size
    image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

    # Convert the resized image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Seed parameters
    seed_distance = 30
    threshold = 5
    min_size = 100
    max_distance = seed_distance // 2
    shrink_pixels = 4

    image_drawn = np.zeros_like(image)

    # Seed the image regularly at a configurable pixel distance grid
    for i in range(0, image.shape[0], seed_distance):
        for j in range(0, image.shape[1], seed_distance):
            seed = (i, j)
            mask = region_growing(
                lab_image, seed, threshold, min_size, max_distance)

            shrunk_mask = shrink_mask(mask, shrink_pixels)

            # Set the color for all pixels in the mask to the average of colors in the image under the mask
            segment_color = np.mean(image[shrunk_mask > 0], axis=0)
            image_drawn[mask > 0] = segment_color

    # Display the result
    cv2.imshow("Segmented Image", image_drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
