import cv2
import numpy as np
import sys
import time
from screeninfo import get_monitors

class ImageProcessor:
    def __init__(self, image_path):
        self.points = []
        self.radius = 5
        self.point_color = (0, 255, 0)  # Green color for points
        self.thickness = 2  # Circle border thickness
        self.line_color = (0, 0, 255)  # Red color for lines
        self.magnify_window_size = 400  # Size of the magnification window
        self.move_mode = False  # Flag to indicate move mode
        self.img = cv2.imread(image_path)
        self.scale = 1
        self.window_name = 'Image'
        self.dragging_point_index = None  # Index of the point being dragged
        self.last_point_index = None  # Index of the last added or dragged point

        if self.img is None:
            raise ValueError("Image not found or empty.")

        self.calculate_scale()
        self.last_mouse_move_time = 0
        self.mouse_move_delay = 0.02  # Delay in seconds

    def calculate_scale(self):
        # Calculate scale based on screen resolution and image size
        monitor = get_monitors()[0]
        screen_res = monitor.width, monitor.height - 200
        self.scale_width = screen_res[0] / self.img.shape[1]
        self.scale_height = screen_res[1] / self.img.shape[0]
        self.scale = min(self.scale_width, self.scale_height)

        if self.scale < 1:
            self.display_img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale)
        else:
            self.display_img = self.img.copy()

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_left_button_down(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.move_mode and self.dragging_point_index is not None:
            self.handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.move_mode:
            self.dragging_point_index = None

        self.update_display_and_magnification()

    def handle_left_button_down(self, x, y):
        if self.move_mode:
            self.move_existing_point(x, y)
        elif len(self.points) < 4:
            self.add_new_point(x, y)

    def move_existing_point(self, x, y):
        distances = [np.sqrt((pt[0] - x / self.scale)**2 + (pt[1] - y / self.scale)**2) for pt in self.points]
        nearest_point_index = np.argmin(distances)
        if distances[nearest_point_index] < 4 * self.radius:
            self.dragging_point_index = nearest_point_index
            self.last_point_index = self.dragging_point_index

    def add_new_point(self, x, y):
        self.points.append((int(x / self.scale), int(y / self.scale)))
        self.last_point_index = len(self.points) - 1
        if len(self.points) == 4:
            self.move_mode = True

    def handle_mouse_move(self, x, y):
        self.points[self.dragging_point_index] = (int(x / self.scale), int(y / self.scale))

    def calculate_perspective_transform(self):
        if len(self.points) != 4:
            return None

        # Calculate the width and height of the undistorted image by summing the widths and heights of the distorted quadrilateral
        width_A_B = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[1]))
        width_C_D = np.linalg.norm(np.array(self.points[2]) - np.array(self.points[3]))
        height_A_D = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[3]))
        height_B_C = np.linalg.norm(np.array(self.points[1]) - np.array(self.points[2]))

        width = int(width_A_B + width_C_D)
        height = int(height_A_D + height_B_C)

        # Define the points for the perspective transform
        src_points = np.float32(self.points)
        dst_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transform to the image
        undistorted_img = cv2.warpPerspective(self.img, M, (width, height))

        return undistorted_img

    def update_display_and_magnification(self):
        # Don't do heavy drawing stuff in call back too often
        current_time = time.time()
        if current_time - self.last_mouse_move_time < self.mouse_move_delay:
            return
        self.last_mouse_move_time = current_time

        # Redraw the display image with updated points
        self.display_img = self.img.copy() if self.scale >= 1 else cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale)
        self.draw_points_and_lines()
        self.show_magnification()

        # Calculate and display the perspective undistorted image
        undistorted_img = self.calculate_perspective_transform()
        if undistorted_img is not None:
            cv2.imshow('Undistorted Image', undistorted_img)

    def draw_points_and_lines(self):
        for point in self.points:
            cv2.circle(self.display_img, (int(point[0] * self.scale), int(point[1] * self.scale)), self.radius, self.point_color, self.thickness, cv2.LINE_AA)
        if len(self.points) == 4:
            for i in range(4):
                cv2.line(self.display_img, (int(self.points[i][0] * self.scale), int(self.points[i][1] * self.scale)), (int(self.points[(i+1)%4][0] * self.scale), int(self.points[(i+1)%4][1] * self.scale)), self.line_color, 1, cv2.LINE_AA)

    def show_magnification(self):
        if self.last_point_index is not None:
            point = self.points[self.last_point_index]
        else:
            return

        magnify_factor = 4
        magnify_size = self.magnify_window_size // magnify_factor
        mx1 = max(point[0] - magnify_size // 2, 0)
        my1 = max(point[1] - magnify_size // 2, 0)
        mx2 = min(point[0] + magnify_size // 2, self.img.shape[1])
        my2 = min(point[1] + magnify_size // 2, self.img.shape[0])

        # Extract the region of interest from the original image
        roi = self.img[my1:my2, mx1:mx2]

        # Resize the extracted region to create the magnified view
        magnify_img = cv2.resize(roi, (self.magnify_window_size, self.magnify_window_size))

        # Draw the crosshair on the magnified view
        cross_color = (255, 255, 255, 128)  # White color with 50% opacity
        cv2.line(magnify_img, (magnify_img.shape[1] // 2, 0), (magnify_img.shape[1] // 2, magnify_img.shape[0]), cross_color, 1)
        cv2.line(magnify_img, (0, magnify_img.shape[0] // 2), (magnify_img.shape[1], magnify_img.shape[0] // 2), cross_color, 1)

        # Show the magnification window
        cv2.imshow('Magnification', magnify_img)

    def main(self):
        cv2.namedWindow(self.window_name)
        cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.draw_circle)

        while True:
            # Display the image
            cv2.imshow(self.window_name, self.display_img)
            k = cv2.waitKey(66)

            if k == 27:  # ESC key to exit
                break

            if self.last_point_index is not None:
                self.handle_key_press(k)

        cv2.destroyAllWindows()

        # Print the coordinates of the points
        if len(self.points) == 4:
            print("Coordinates of the quadrilateral:")
            print(" ".join(f"{point[0]},{point[1]}" for point in self.points))

    def handle_key_press(self, k):
        dx, dy = 0, 0
        if k == 52:  # 4 numkey
            dx = -1
        elif k == 56:  # 8 num key
            dy = -1
        elif k == 54:  # 6 numkey
            dx = 1
        elif k == 50:  # 2 numkey
            dy = 1

        if dx != 0 or dy != 0:
            x, y = self.points[self.last_point_index]
            self.points[self.last_point_index] = (x + dx, y + dy)
            self.update_display_and_magnification()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        processor = ImageProcessor(sys.argv[1])
        processor.main()
    else:
        print("Usage: python script.py path_to_image")
