# IMPORTS
import cv2
import numpy

# Formats
IMG_FORMATS = ['png', 'jpg', 'jpeg', 'tiff']
VIDEO_FORMATS = ['mp4', 'aac']
# Detectors
DETECTORS = ['SIFT', 'SURF', 'ORB']


def detectFeatures (path_to_img: str, detector: str, color_correction: bool) -> numpy.ndarray:
    """
    Given :param path_to_img: performs feature detection based on
    the specified :param detector:
    :param path_to_img: path to image to be processed
    :param detector: supported feature detectors {`SIFT`, `SURF`, `ORB`}
    :return: processed image
    """

    if detector:
        detector = detector.lower()
        match detector:
            case 'sift':
                model = cv2.xfeatures2d.SIFT_create()
            case 'surf':
                model = cv2.xfeatures2d.SURF_create()
            case 'orb':
                model = cv2.ORB_create()
            case _:
                raise NotImplementedError(f'{detector} has not yet been implemented!\n'
                                          f'Please choose between {DETECTORS}!')
    else:
        raise f"Please choose a detector between {DETECTORS}!"

    if color_correction:
        img = cv2.imread(path_to_img, cv2.COLOR_BGR2GRAY)   # Apply Color Correction
    else:
        img = cv2.imread(path_to_img)                       # Reading Image

    kp, descriptors = model.detectAndCompute(img, None)     # Saving KeyPoints
    img_out = cv2.drawKeypoints(img, kp, None)              # Spit Out Image
    return img_out

