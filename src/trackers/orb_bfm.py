import cv2
from src.matchers import Matchers
from src.detectors import Detectors
import src.utils.color_coding as cc
from src.utils.GLOBALS import MATCH_COLOR, KP_COLOR

orb = cv2.ORB_create()
bf = cv2.BFMatcher()

cap = cv2.VideoCapture('../../data/Contesto_industriale1.mp4')

prev_kp = None
prev_des = None
frames_count = 0    # init frame_count
num_detections = 0  # init num_detections

while cap.isOpened():

    ret, frame = cap.read()
    frame = cc.make_BW(frame)
    kp, des = orb.detectAndCompute(frame, None)

    if prev_kp is None:
        pass
    else:
        matches = bf.knnMatch(des, prev_des, k=2)

        # Draw all detected keypoints
        frame = cv2.drawKeypoints(frame, kp, outImage=frame, color=KP_COLOR)
        prev_frame = cv2.drawKeypoints(prev_frame, prev_kp, outImage=prev_frame, color=KP_COLOR)

        good_matches = []
        for i, j in matches:
            if i.distance < 0.3 * j.distance:
                good_matches.append([i])

        img = cv2.drawMatchesKnn(
            frame, kp,
            prev_frame, prev_kp,
            good_matches, None,
            matchColor=MATCH_COLOR,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow('ORB + BFM', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frames_count += 1
        num_detections = len(good_matches)

    prev_kp = kp
    prev_des = des
    prev_frame = frame

cap.release()
cv2.destroyAllWindows()
