import cv2
import matplotlib.pyplot as plt
import src.utils.feature_detection as fd


cap = cv2.VideoCapture("../../data/Contesto_industriale1.mp4")

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

last_frame_features = None
last_frame_image = None

descriptor = fd.get_detector('sift', 1000)
matcher = fd.get_matcher('bf', cv2.NORM_L2)

while cap.isOpened():

    ret, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp, des = descriptor.detectAndCompute(gray_image, None)
    print(des)
    if last_frame_features is not None and des is not None:
        matches = matcher.knnMatch(last_frame_features, des, k=2)

        # Ratio Test
        # good_matches = fd.find_good_matches_knn(
        #     matcher=matcher,
        #     des1=last_frame_features,
        #     des2=des)
        # Lowe's Ratio Test
        good_matches = fd.find_good_matches_Lowe(matches)
        draw_params = dict(singlePointColor=(255, 0, 0),
                           matchesMask=good_matches,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img = fd.draw_good_matches_Lowe(
            last_frame_image, last_frame_kp,
            frame, kp,
            matches, good_matches, draw_params)

        # img = fd.draw_good_matches_knn(
        #     last_frame_image, last_frame_kp,
        #     frame, kp,
        #     good_matches,
        # )
        # img = cv2.drawMatchesKnn(
        #     last_frame_image, last_frame_kp,
        #     frame, kp,
        #     good_matches, None, **draw_params)
        cv2.imshow('Lowe', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    last_frame_features = des
    last_frame_kp = kp
    last_frame_image = frame

cap.release()
cv2.destroyAllWindows()
