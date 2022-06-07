import os.path
import cv2

# GLOBALS
PATH_TO_OBJ = os.path.join(os.getcwd(), 'data', 'TrainMeter.png')
PATH_TO_SCENE = os.path.join(os.getcwd(), 'data', 'Meters.png')
NUM_FEATURES = 500

# Images w/ Color Correction
img_obj = cv2.imread(PATH_TO_OBJ, cv2.COLOR_BGR2GRAY)
img_scene = cv2.imread(PATH_TO_SCENE, cv2.COLOR_BGR2GRAY)

# INITS:
# Sift
sift = cv2.SIFT_create(NUM_FEATURES)
# Feature Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# KeyPoints & Descriptors
kp_obj, des_obj = sift.detectAndCompute(img_obj, None)
kp_scene, des_scene = sift.detectAndCompute(img_obj, None)

# Matches
# matches = bf.match(des_obj, des_scene)
# all_matches = sorted(matches, key=lambda x: x.distance)
# good_matches = list(filter(lambda x: x.distance < 150, matches))
#
# all_results = cv2.drawMatches(
#     img_obj, kp_obj,
#     img_scene, kp_scene,
#     all_matches,
#     img_scene,
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )
#
# good_results = cv2.drawMatches(
#     img_obj, kp_obj,
#     img_scene, kp_scene,
#     good_matches,
#     img_scene,
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )

# Showing Results
cv2.imshow('OBJ', img_obj)
cv2.imshow('SCENE', img_scene)
#cv2.imshow('ALL', all_results)
#cv2.imshow('GOOD', good_results)
cv2.waitKey(0)
