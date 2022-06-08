import cv2

PATH_TO_OBJ = '../data/TrainMeter.png'
PATH_TO_SCENE = '../data/Meters.png'
PATH_TO_VIDEO = '../data/Contesto_industriale1.mp4'
NUM_FEATURES = 400

# SIFT
sift = cv2.SIFT_create(NUM_FEATURES)

# FEATURE MATCHING
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Reading Scene and Object images
scene_img = cv2.imread(PATH_TO_SCENE)
obj_img = cv2.imread(PATH_TO_OBJ)
# Color Correction
scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(PATH_TO_VIDEO)
while cap.isOpened():
    ret, frame = cap.read()
    sift_img = frame.copy()
    # Key points, Descriptors
    kp_1, des_1 = sift.detectAndCompute(sift_img, None)
    kp_2, des_2 = sift.detectAndCompute(obj_img, None)

    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key=lambda x: x.distance)


    # All Matches
    all_matches = cv2.drawMatches(
        sift_img, kp_1,
        obj_img, kp_2,
        matches,
        obj_img,
        flags=2)
    # Only 600 Matches
    some_matches = cv2.drawMatches(
        sift_img, kp_1,
        obj_img, kp_2,
        matches[:600],
        obj_img,
        flags=2)
    # Only matches with distance smaller than 150
    good_matches = cv2.drawMatches(
        sift_img, kp_1,
        obj_img, kp_2,
        list(filter(lambda x: x.distance < 150, matches)),
        obj_img,
        flags=2)

    cv2.imshow('ALL', all_matches)
    cv2.imshow('SOME', some_matches)
    cv2.imshow('GOOD', good_matches)
    if cv2.waitKey(1) == ord('q'):
        break
