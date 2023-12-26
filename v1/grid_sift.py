import cv2
import sys
import matplotlib.pyplot as plt

def extract_sift_features(image, block_size, response_threshold):
    sift = cv2.SIFT_create(nfeatures=32)

    keypoint_list = []
    height, width = image.shape[:2]

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            keypoints, _ = sift.detectAndCompute(block, None)
            # Filter keypoints based on response threshold
            keypoints = [kp for kp in keypoints if kp.response > response_threshold]
            for kp in keypoints:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
            keypoint_list.extend(keypoints)

    return keypoint_list

def match_and_display(img1, img2, block_size, response_threshold):
    kp1 = extract_sift_features(img1, block_size, response_threshold)
    kp2 = extract_sift_features(img2, block_size, response_threshold)

    img1_sift = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_sift = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Match SIFT features globally
    matcher = cv2.BFMatcher()
    matches = matcher.match(cv2.SIFT_create().compute(img1, kp1)[1], cv2.SIFT_create().compute(img2, kp2)[1])

    # Draw matches on a new image
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow('Image 1 SIFT Features', img1_sift)
    # cv2.imshow('Image 2 SIFT Features', img2_sift)
    # cv2.imshow('SIFT Matches', match_img)

    fig, ax = plt.subplots()
    # ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    # plt.show()
    ax.imshow(cv2.cvtColor(img1_sift, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py image1_path image2_path")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Could not read images.")
        sys.exit(1)

    block_size = 256

    response_threshold = 0.08
    match_and_display(img1, img2, block_size, response_threshold)
