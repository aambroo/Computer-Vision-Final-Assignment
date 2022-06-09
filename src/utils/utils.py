# IMPORTS
import cv2
import numpy

# Formats
IMG_FORMATS = ['png', 'jpg', 'jpeg', 'tiff']
VIDEO_FORMATS = ['mp4', 'aac']
# Detectors
DETECTORS = ['SIFT', 'SURF', 'ORB']


def detectFeatures(path_to_img: str, detector: str, color_correction: bool, num_features: int) -> numpy.ndarray:
    """
    Given :param path_to_img: performs feature detection based on
    the specified :param detector:
    :param path_to_img: path to image to be processed
    :param detector: supported feature detectors {`SIFT.ipynb`, `SURF`, `ORB`}
    :return: processed image
    """

    if detector:
        detector = detector.lower()
        match detector:
            case 'sift':
                if num_features:
                    return cv2.xfeatures2d.SIFT_create(num_features)
                else:
                    return cv2.xfeatures2d.SIFT_create()
            case 'surf':
                if num_features:
                    return cv2.xfeatures2d.SURF_create(num_features)
                else:
                    return cv2.xfeatures2d.SURF_create()
            case 'orb':
                if num_features:
                    return cv2.ORB_create(num_features)
                else:
                    return cv2.ORB_create()
            case _:
                raise NotImplementedError(f'{detector} has not yet been implemented!\n'
                                          f'Please choose between {DETECTORS}!')
    else:
        raise f"Please choose a detector between {DETECTORS}!"

    if color_correction:
        img = cv2.imread(path_to_img, cv2.COLOR_BGR2GRAY)  # Apply Color Correction
    else:
        img = cv2.imread(path_to_img)  # Reading Image

    kp, descriptors = model.detectAndCompute(img, None)  # Saving KeyPoints
    img_out = cv2.drawKeypoints(img, kp, None)  # Spit Out Image
    return img_out


def drawKeyPoints(img, key_points, out_image):
    """
    :param img: src image of key points
    :param key_points: image's key points
    :param out_image: image to draw key points on
    :return: :param out_image: with :param key_points: drawn on
    """
    return cv2.drawKeypoints(img, key_points, out_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def create_model(detector: str, num_features: int):
    detector = detector.lower()
    match detector:
        case 'sift':
            if num_features:
                return cv2.SIFT_create(num_features)
            else:
                return cv2.SIFT_create()
        case 'akaze':
            return cv2.AKAZE_create()
        case 'orb':
            if num_features:
                return cv2.ORB_create(num_features)
            else:
                return cv2.ORB_create()
        case _:
            raise NotImplementedError(f'{detector} has not yet been implemented!\n'
                                      f'Please choose between {DETECTORS}!')