import cv2
from src.matchers import Matchers
from src.detectors import Detectors
import src.utils.color_coding as cc

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

cap = cv2.VideoCapture('../../data/Contesto_industriale1.mp4')

prev_kp = None
prev_des = None

while cap.isOpened():

    ret, frame = cap.read()
    frame = cc.make_BW(frame)
    kp, des = sift.detectAndCompute(frame, None)

    if prev_kp is None:
        pass
    else:
        matches = bf.knnMatch(des, prev_des, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good_matches.append([m])

        img = cv2.drawMatchesKnn(
            frame, kp,
            prev_frame, prev_kp,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    prev_kp = kp
    prev_des = des
    prev_frame = frame

cap.release()
cv2.destroyAllWindows()
