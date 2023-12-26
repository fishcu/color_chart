import sys
import rawpy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def display_16bit_image(image_path):
    try:
        # Open the raw image file
        with rawpy.imread(image_path) as raw:
            # Process the raw image with specified settings
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)

            # Normalize pixel values to the valid range [0, 1]
            norm = Normalize(vmin=rgb.min(), vmax=rgb.max())
            normalized_rgb = norm(rgb)

            # Display the 16-bit image using matplotlib
            plt.imshow(normalized_rgb)
            plt.axis('off')  # Turn off axis labels
            plt.show()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python display_16bit_image.py <image_path>")
    else:
        # Get the image path from command line arguments
        image_path = sys.argv[1]
        display_16bit_image(image_path)
