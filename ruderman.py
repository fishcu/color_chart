import numpy as np

def rgb_to_lab(input_image):
    # Transformation matrices for converting RGB to LMS and LMS to Lab
    rgb_to_lms = np.array([[0.3811, 0.5783, 0.0402],
                           [0.1967, 0.7244, 0.0782],
                           [0.0241, 0.1288, 0.8444]], dtype=np.float32)

    lms_to_lab = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                           [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
                           [1/np.sqrt(2), -1/np.sqrt(2), 0]], dtype=np.float32)

    # Convert input image to float32 for numerical precision.
    rgb_image = input_image.astype(np.float32)

    # Apply the first stage transformation (RGB to LMS).
    lms_image = np.dot(rgb_image.reshape(-1, 3), rgb_to_lms.T)

    # Ensure that all values are above a small epsilon to avoid issues in logarithmic transformations.
    epsilon = 1.0/255.0
    lms_image = np.maximum(lms_image, epsilon)

    # Apply logarithmic transformation.
    lms_image = np.log10(lms_image)

    # Apply the second stage transformation (LMS to Lab).
    lab_image = np.dot(lms_image, lms_to_lab.T).reshape(input_image.shape)

    return lab_image
