import cv2
from src.utils.GLOBALS import *


def get_detector(detector: str, num_features: int):
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
                                      f'Please choose one between {DETECTORS}!')


def get_matcher(matcher: str, *args):
    matcher = matcher.lower()
    match matcher:
        case 'bf':
            if args:
                return cv2.BFMatcher(*args)
            else:
                return cv2.BFMatcher()
        case 'flann':
            if args:
                return cv2.FlannBasedMatcher(*args)
            else:
                return cv2.FlannBasedMatcher()
        case _:
            raise NotImplementedError(f'{matcher} has not yet been implemented!\n'
                                      f'Please choose one between {MATCHERS}!')
