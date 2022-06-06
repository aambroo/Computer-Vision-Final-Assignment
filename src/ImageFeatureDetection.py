from utils import imgDetectFeatures
import cv2


path_to_img = '../data/Meters.png'
# img = cv2.imread(path_to_img)

# surf = cv2.SIFT_create()
# kp, descriptors = surf.detectAndCompute(img, None)
# img = cv2.drawKeypoints(img, kp, None)
# print(type(img))


img = imgDetectFeatures(path_to_img, 'ORB')
cv2.imshow('ORB', img)
cv2.waitKey(0)
cv2.destroyAllWindows()