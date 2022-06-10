import cv2
from src.utils.GLOBALS import SIFT_INDEX_PARAMS, SIFT_SEARCH_PARAMS, MRT, KP_COLOR, MATCH_COLOR
import src.utils.color_coding as cc

akaze = cv2.AKAZE_create()
flann = cv2.FlannBasedMatcher(SIFT_INDEX_PARAMS, SIFT_SEARCH_PARAMS)

cap = cv2.VideoCapture('../../data/Contesto_industriale1.mp4')

prev_kp = None
prev_des = None
frames_count = 0    # init frame_count
num_detections = 0  # init num_detections

while cap.isOpened():

    ret, frame = cap.read()
    frame = cc.make_BW(frame)
    kp, des = akaze.detectAndCompute(frame, None)

    if prev_kp is None:
        pass
    else:
        matches = flann.knnMatch(des, prev_des, k=2)

        # Use mask to only draw good matches -> replaces good_matches list
        matches_mask = [[0, 0] for i in range(len(matches))]

        # Addresses an issue due to possible miss-pairing of matches
        for idx, pair in enumerate(matches):
            try:
                i, j = pair
            except:
                continue
            if i.distance < MRT * j.distance:
                matches_mask[idx] = [1, 0]

        good_matches = []
        for i, j in matches:
            if i.distance < MRT * j.distance:
                good_matches.append([i])

        draw_params = dict(
            matchColor=MATCH_COLOR,
            singlePointColor=KP_COLOR,
            matchesMask=matches_mask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img = cv2.drawMatchesKnn(
            frame, kp,
            prev_frame, prev_kp,
            good_matches, None,
            **draw_params
        )
        cv2.imshow('AKAZE + FLANN', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frames_count += 1
        num_detections = len(matches_mask)

    prev_kp = kp
    prev_des = des
    prev_frame = frame

cap.release()
cv2.destroyAllWindows()
