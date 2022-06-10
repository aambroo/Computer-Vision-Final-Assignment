import cv2
from src.utils.GLOBALS import ORB_SEARCH_PARAMS, ORB_INDEX_PARAMS, MRT, MATCH_COLOR, KP_COLOR
import src.utils.color_coding as cc

orb = cv2.ORB_create()
flann = cv2.FlannBasedMatcher(ORB_INDEX_PARAMS, ORB_SEARCH_PARAMS)

cap = cv2.VideoCapture('../../data/Contesto_industriale1.mp4')

prev_kp = None
prev_des = None
frames_count = 0  # init frame_count
num_detections = 0  # init num_detections

while cap.isOpened():

    ret, frame = cap.read()
    frame = cc.make_BW(frame)
    kp, des = orb.detectAndCompute(frame, None)

    if prev_kp is None:
        pass
    else:
        matches = flann.knnMatch(prev_des, des, k=2)

        # Use mask to only draw good matches -> replaces good_matches list
        matches_mask = [[0, 0] for _ in range(len(matches))]

        # Addresses an issue due to possible miss-pair of matches
        for idx, pair in enumerate(matches):
            try:
                i, j = pair
            except:
                continue
            if i.distance < MRT * j.distance:
                matches_mask[idx] = [1, 0]

        draw_params = dict(
            matchColor=MATCH_COLOR,
            singlePointColor=KP_COLOR,
            matchesMask=matches_mask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img = cv2.drawMatchesKnn(
            prev_frame, prev_kp,
            frame, kp,
            matches, None,
            matchesMask=matches_mask
        )
        cv2.imshow('ORB + FLANN', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frames_count += 1
        num_detections = len(matches_mask)

    prev_kp = kp
    prev_des = des
    prev_frame = frame

cap.release()
cv2.destroyAllWindows()
