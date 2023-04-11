from glob import glob
import os
import re
import cv2
import numpy as np
from PIL import Image


def removing_files() -> None:
    """
    removing duplicate images
    :return: None
    """
    pattern = r"\((\d+)\)"
    os.makedirs('new_data', exist_ok=True)
    files_paths = glob('data/**/**.png', recursive=True)
    for file in files_paths:
        match = re.search(pattern, file)
        if match:
            os.remove(file)


def remove_blob_aux(im_path: str) -> None:
    """
    removing large object from the image using connected components
    :param im_path:
    :return:
    """
    # Load the input image as grayscale
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    # Threshold the image to create a binary image
    _, binary_img = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # Find connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

    # Set a threshold for the minimum size of the connected components to keep
    min_size = 200

    # Create a mask to store the pixels to keep
    mask = np.zeros_like(img)

    # Iterate over the connected components and keep only the ones that are smaller than the threshold
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 255

    # Apply the mask to the original image to remove the large objects
    result = cv2.bitwise_and(img, mask)
    result = cv2.bitwise_not(result)
    return result


def remove_blobs() -> None:
    """
    removing noise from the images
    :return: None
    """
    files_paths = glob('data/**/**.png', recursive=True)
    for file in files_paths:
        folder = file.split('/')[-2]
        os.makedirs(f'preprocessed_data/{folder}', exist_ok=True)
        im_name = file.split('/')[-1]
        rs = remove_blob_aux(file)
        cv2.imwrite(f'preprocessed_data/{folder}/{im_name}', rs)


def extract_points_from_im(path: str) -> (np.array, np.array):
    """
    reading a image file, extracting from the image coordinates
    :param path: str
    :return:
    """
    image = Image.open(path)
    image = image.convert('L')
    coordinates = []
    x_values = []
    y_values = []
    width, height = image.size

    # Iterate over every pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the pixel value at (x, y)
            pixel = image.getpixel((x, y))
            # Check if the pixel is not white
            if pixel != 255:
                # Add the coordinates of the non-white pixel to the list
                coordinates.append((x, y))
                x_values.append(x)
                y_values.append(height - y)

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    return x_values, y_values


def avg_noise(x: list, y: list) -> (int, int):
    """
    for point who has the same x value aggregate the y value using average
    :param x: list of x coordinate
    :param y: list of y coordinate
    :return:
    """
    points = {}
    for x_cord, y_cord in zip(x, y):
        if x_cord in points:
            points[x_cord].append(y_cord)
        else:
            points[x_cord] = [y_cord]

    for key, value in points.items():
        points[key] = np.average(value)

    x = np.array(list(points.keys()))
    y = np.array(list(points.values()))
    return x, y


def main():
    remove_blobs()


if __name__ == '__main__':
    main()
