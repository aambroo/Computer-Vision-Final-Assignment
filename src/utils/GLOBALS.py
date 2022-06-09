import os

# HyperParameters
NUM_FEATURES = 50

# DATA PATHS
PATH_TO_OBJ = os.path.join(os.getcwd(), 'data', 'Contatore.png')
PATH_TO_SCENE = os.path.join(os.getcwd(), 'data', 'Meters.png')
PATH_TO_VIDEO = os.path.join(os.getcwd(), 'data', 'Contesto_industriale1.mp4')

# MISCELLANEOUS
DETECTORS = ['SIFT', 'AKAZE', 'ORB']
MATCHERS = ['BF', 'FLANN']
