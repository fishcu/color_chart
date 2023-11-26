import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
import sys
import os

def load_rgb_matrix(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        rgb_matrix = np.array([list(map(int, line.strip().split())) for line in lines], dtype=int)
        return rgb_matrix
    except Exception as e:
        print(f"Error loading RGB matrix from {file_path}: {e}")
        return None
    
def rgb_to_lab(rgb):
    rgb_normalized = rgb / 255.0
    lab = rgb2lab(rgb_normalized.reshape(1, 1, 3))
    return lab[0, 0, :]

def calculate_delta_e(color1, color2):
    return deltaE_ciede2000(color1, color2)

def calculate_rms_delta_e(delta_e_scores):
    return np.sqrt(np.mean(delta_e_scores**2))

def compute_delta_e_values(rgb_matrix_1, rgb_matrix_2):
    delta_e_scores = []
    for i in range(rgb_matrix_1.shape[0]):
        lab_1 = rgb_to_lab(rgb_matrix_1[i, :])
        lab_2 = rgb_to_lab(rgb_matrix_2[i, :])
        delta_e = calculate_delta_e(lab_1, lab_2)
        delta_e_scores.append(delta_e)

    # Append RMS delta E to the list
    rms_delta_e = calculate_rms_delta_e(np.array(delta_e_scores))
    delta_e_scores.append(rms_delta_e)

    return delta_e_scores

def save_delta_e_results(output_file, delta_e_scores, decimal_places=3):
    with open(output_file, 'w') as file:
        file.write(f"Sample,DeltaE\n")
        for i, delta_e in enumerate(delta_e_scores[:-1]):  # Exclude the last element (RMS)
            rounded_delta_e = round(delta_e, decimal_places)
            file.write(f"{i+1},{rounded_delta_e}\n")
        rounded_rms_delta_e = round(delta_e_scores[-1], decimal_places)
        file.write(f"RMS,{rounded_rms_delta_e}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <ref_rgb_matrix_file> <test_rgb_matrix_file>")
        sys.exit(1)

    # Load RGB matrices
    rgb_matrix_1 = load_rgb_matrix(sys.argv[1])
    rgb_matrix_2 = load_rgb_matrix(sys.argv[2])

    if rgb_matrix_1 is None or rgb_matrix_2 is None:
        sys.exit(1)

    # Compute delta E values
    delta_e_scores = compute_delta_e_values(rgb_matrix_1, rgb_matrix_2)

    # Save delta E results to a file
    output_file = os.path.splitext(sys.argv[2])[0] + "_delta_e.txt"
    save_delta_e_results(output_file, delta_e_scores)

    print(f"Results saved to {output_file}")

    print(round(delta_e_scores[-1], 3))

if __name__ == "__main__":
    main()