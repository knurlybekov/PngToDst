import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyembroidery import EmbPattern, write_dst, STITCH, JUMP, TRIM
import os

def read_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    return edges

def find_contours(edges):
    # Find contours and hierarchy in the edged image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def create_stitch_pattern(contours, hierarchy):
    pattern = EmbPattern()
    stitch_length = 5

    # Flatten hierarchy array for easier indexing
    hierarchy = hierarchy[0]

    def process_contour(index, level):
        contour = contours[index]

        # Skip small contours to reduce stitch count
        if cv2.arcLength(contour, True) < 10:
            return

        # Resample contour to control the number of stitches
        contour_length = cv2.arcLength(contour, True)
        num_points = max(int(contour_length / stitch_length), 1)
        resampled_contour = resample_contour(contour, num_points)

        # Determine stitching direction based on level
        if level % 2 == 0:
            # Even level: Outer contour (stitch clockwise)
            pass
        else:
            # Odd level: Hole (stitch counter-clockwise)
            resampled_contour = resampled_contour[::-1]

        # Move to the starting point
        x, y = resampled_contour[0][0]
        pattern.add_stitch_absolute(JUMP, x, y)

        # Stitch along the contour
        for point in resampled_contour[1:]:
            x, y = point[0]
            pattern.add_stitch_absolute(STITCH, x, y)

        # Close the contour
        x, y = resampled_contour[0][0]
        pattern.add_stitch_absolute(STITCH, x, y)

        # Trim thread after each contour
        pattern.add_command(TRIM)

        # Process child contours (holes)
        child = hierarchy[index][2]
        while child != -1:
            process_contour(child, level + 1)
            child = hierarchy[child][0]

    # Start processing from the outermost contours
    for i in range(len(contours)):
        # Process only top-level contours (no parent)
        if hierarchy[i][3] == -1:
            process_contour(i, level=0)

    pattern.end()
    return pattern

# Helper Function to Resample Contour
def resample_contour(contour, num_points):
    # Create a list of points from the contour
    contour_points = contour[:, 0, :]  # Shape: (N, 2)

    # Calculate cumulative distances between points
    deltas = np.diff(contour_points, axis=0)
    segment_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))  # Shape: (N,)

    # Total length of the contour
    total_length = cumulative_lengths[-1]

    # New distances at which to sample the contour
    distances = np.linspace(0, total_length, num_points)

    # Interpolate x and y separately
    x_interp = np.interp(distances, cumulative_lengths, contour_points[:, 0])
    y_interp = np.interp(distances, cumulative_lengths, contour_points[:, 1])

    # Combine interpolated x and y coordinates
    resampled_points = np.vstack((x_interp, y_interp)).T.astype(np.int32)
    resampled_points = resampled_points.reshape(-1, 1, 2)  # Shape: (num_points, 1, 2)

    return resampled_points

def save_pattern(pattern, output_file):
    write_dst(pattern, output_file)
    print(f"Stitch pattern saved as {output_file}")

# Main Workflow
if __name__ == "__main__":
    image_path = 'bird.jpg'  # Path to your image in png or jpg
    dst_file = 'output.dst'

    image = read_image(image_path)

    preprocessed_image = preprocess_image(image)

    edges = detect_edges(preprocessed_image)

    # Display detected edges
    plt.figure(figsize=(8, 6))
    plt.imshow(edges, cmap='gray')
    plt.title('Detected Edges')
    plt.axis('off')
    plt.show()

    contours, hierarchy = find_contours(edges)

    # Display the contour image
    contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.axis('off')
    plt.show()

    stitch_pattern = create_stitch_pattern(contours, hierarchy)

    save_pattern(stitch_pattern, dst_file)
