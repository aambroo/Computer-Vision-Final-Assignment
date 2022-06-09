import cv2
import src.utils.feature_detectors as fd
import src.utils.color_coding as cc

DETECTOR_NAME = 'AKAZE'

detector = fd.get_detector(DETECTOR_NAME, None)
video = cv2.VideoCapture('../../data/Contesto_industriale1.mp4')

while video.isOpened():
    ret, frame = video.read()
    grayscale_img = cc.make_BW(frame)

    kp = detector.detect(grayscale_img, None)
    frame = cv2.drawKeypoints(frame, kp, None, color=(180, 0, 180))

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
