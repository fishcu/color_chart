import cv2
import numpy as np
import sys

from skimage import color

class KodakExtractor:
    def __init__(self, points):
        self.points = points
        self.cell_ratio = 0.8  # Ratio of cell to be filled by rectangle
        self.num_cells_x = 19  # Number of cells in x direction
        self.num_cells_y = 12  # Number of cells in y direction

        # Calculate the width and height of the undistorted image by summing the widths and heights of the distorted quadrilateral
        width_A_B = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[1]))
        width_C_D = np.linalg.norm(np.array(self.points[2]) - np.array(self.points[3]))
        height_A_D = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[3]))
        height_B_C = np.linalg.norm(np.array(self.points[1]) - np.array(self.points[2]))

        self.out_width = int(width_A_B + width_C_D)
        self.out_height = int(height_A_D + height_B_C)

        # Define the points for the perspective transform
        src_points = np.float32(self.points)
        dst_points = np.float32([[0, 0], [self.out_width - 1, 0], [self.out_width - 1, self.out_height - 1], [0, self.out_height - 1]])

        # Calculate the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)

    def calculate_perspective_transform(self, img):
        # Apply the perspective transform to the image
        undistorted_img = cv2.warpPerspective(img, self.M, (self.out_width, self.out_height))

        return undistorted_img

    def divide_into_cells_and_draw_rectangles(self, undistorted_img):
        cell_width = undistorted_img.shape[1] / self.num_cells_x
        cell_height = undistorted_img.shape[0] / self.num_cells_y
        median_colors = np.zeros(((self.num_cells_x - 1) * self.num_cells_y, 3))
        num_proc = 0

        for i in range(self.num_cells_x):
            if i == 1:  # Skip the 2nd column
                continue
            for j in range(self.num_cells_y):
                top_left = (int(i * cell_width), int(j * cell_height))
                bottom_right = (int((i + 1) * cell_width), int((j + 1) * cell_height))

                # Calculate the rectangle dimensions based on the cell dimensions and the desired ratio
                # rect_width = cell_width * self.cell_ratio
                # rect_height = cell_height * self.cell_ratio

                # Calculate the top left and bottom right points for the rectangle
                # rect_top_left = (int(top_left[0] + (cell_width - rect_width) / 2), int(top_left[1] + (cell_height - rect_height) / 2))
                # rect_bottom_right = (int(rect_top_left[0] + rect_width), int(rect_top_left[1] + rect_height))

                # Draw the rectangle on the image
                # cv2.rectangle(undistorted_img, rect_top_left, rect_bottom_right, (0, 255, 0), 1)

                # Extract the cell region
                cell_img = undistorted_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                # Calculate the median color of the cell
                median_color = np.median(cell_img.reshape(-1, 3), axis=0)
                # Swap BGR -> RGB
                tmp = median_color[0]
                median_color[0] = median_color[2]
                median_color[2] = tmp
                median_colors[num_proc] = median_color
                num_proc += 1

        return undistorted_img, median_colors

    def get_rgb_values(self, img):
        if img is None:
            raise ValueError("Image not found or empty.")
        undistorted_img = self.calculate_perspective_transform(img)
        result_img, median_colors = self.divide_into_cells_and_draw_rectangles(undistorted_img)
        # cv2.imshow('Result Image', result_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return median_colors
    
    def get_lab_values(self, img):
        rgb = self.get_rgb_values(img)
        return color.rgb2lab(rgb / 255.0)
    

def draw_cell_grid(cell_colors, num_cells_x, num_cells_y, cell_size, skip_column=1, skip_color=[128, 128, 128]):
    # Create a blank image with the size to fit the grid
    img_height = num_cells_y * cell_size
    img_width = num_cells_x * cell_size
    grid_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    color_index = 0
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            # Calculate the top left corner of the cell
            top_left_x = i * cell_size
            top_left_y = j * cell_size

            # Fill the cell with the specified color or the skip color for the 2nd column
            color = skip_color if i == skip_column else cell_colors[color_index]
            tmp = color[0]
            color[0] = color[2]
            color[2] = tmp
            cv2.rectangle(grid_image, (top_left_x, top_left_y), (top_left_x + cell_size, top_left_y + cell_size), color, -1)

            # Only increment the color index if we're not in the skip column
            if i != skip_column:
                color_index += 1

    # Draw the grid lines
    for i in range(1, num_cells_x):
        cv2.line(grid_image, (i * cell_size, 0), (i * cell_size, img_height), (255, 255, 255), 1)
    for j in range(1, num_cells_y):
        cv2.line(grid_image, (0, j * cell_size), (img_width, j * cell_size), (255, 255, 255), 1)

    return grid_image


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        points = [tuple(map(int, point.split(','))) for point in sys.argv[2:]]
        img = cv2.imread(image_path)
        processor = KodakExtractor(points)
        median_colors = processor.get_rgb_values(img)
        print("Median colors of each cell:")
        for color in median_colors:
            print(color)

        grid_image = draw_cell_grid(median_colors, 19, 12, 50)
        cv2.imshow('Cell Grid', grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Usage: python script.py path_to_image point1 point2 point3 point4")