import cv2
import src.utils.color_coding as cc
from src.utils.GLOBALS import MATCH_COLOR, KP_COLOR

akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher()

cap = cv2.VideoCapture("../../data/Contesto_industriale1.mp4")

prev_kp = None
prev_des = None

while cap.isOpened():

    ret, frame = cap.read()
    frame = cc.make_BW(frame)
    kp, des = akaze.detectAndCompute(frame, None)

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
        cv2.imshow('AKAZE + BFM', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    prev_kp = kp
    prev_des = des
    prev_frame = frame

cap.release()
cv2.destroyAllWindows()
