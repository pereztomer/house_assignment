import numpy as np
from PIL import Image
import cv2


def main():
    # open method used to open different extension image file
    im = Image.open("data/line/line_00000.png")
    print(im.size)
    np_im = np.array(im)
    # This method will show image in any image viewer
    im.show()


def morf():
    im = cv2.imread('data/line/line_00000.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(im, kernel, iterations=4)
    print('hi')


if __name__ == '__main__':
    morf()
