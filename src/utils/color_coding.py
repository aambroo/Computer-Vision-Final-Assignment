import cv2
import numpy as np


def imageResize(image):
    """
    Resizes the Image to a new aspect ratio
    :param image: image to resize
    :return: resized image
    """
    maxD = 1024
    height, width = image.shape
    aspectRatio = width / height
    if aspectRatio < 1:
        new_size = (int(maxD * aspectRatio), maxD)
    else:
        new_size = (maxD, int(maxD / aspectRatio))
    image = cv2.resize(image, new_size)
    return image


def make_BW(image):
    """
    Makes an image (or list of images) B&W and resizes it
    :param image: image to turn B&W
    :return: B&W :param image:
    """
    if isinstance(image, list):
        imageBW = []
        for imageName in image:
            imagePath = "../data/image/" + str(imageName)
            imageBW.append(imageResize(cv2.imread(imagePath, cv2.COLOR_BGR2GRAY)))
        return imageBW
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # imagePath = '../data/' + str(image)
        return cv2.imread(image, cv2.COLOR_BGR2GRAY)
