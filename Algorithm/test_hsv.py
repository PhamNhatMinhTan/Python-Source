import cv2
import numpy as np
from Matplotlib import *


def nothing(x):
    pass


# create Trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H_min", "Trackbars", 98, 179, nothing)
cv2.createTrackbar("S_min", "Trackbars", 45, 255, nothing)
cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)

cap = cv2.VideoCapture("/home/tan/Python Source/video/test17.mp4")

while True:
    # while cap.isOpened():
    # read image
    # image = cv2.imread("/home/tan/Python Source/video/real/img2_test.jpg")
    image = cv2.imread("/home/tan/Python Source/video/test17_img.png")  # Image test
    # ret, image = cap.read()

    # resize image
    image = cv2.resize(image, (320, 240))

    # converting frame(image == BGR) to HSV(hue-saturation-value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv = show_hsv_equalized(image)

    # get value Trackbars
    H_min = cv2.getTrackbarPos("H_min", "Trackbars")
    S_min = cv2.getTrackbarPos("S_min", "Trackbars")
    V_min = cv2.getTrackbarPos("V_min", "Trackbars")
    H_max = cv2.getTrackbarPos("H_max", "Trackbars")
    S_max = cv2.getTrackbarPos("S_max", "Trackbars")
    V_max = cv2.getTrackbarPos("V_max", "Trackbars")

    # range color
    lower_red = np.array([H_min, S_min, V_min])
    upper_red = np.array([H_max, S_max, V_max])

    # Morphological Transformations,Opening and Closing
    thresh = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(image, image, mask=mask)
    # result = cv2.Canny(thresh, 50, 100)
    result = cv2.Canny(mask, 50, 100)
    cv2.imshow("Original", image)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(1)
    if key == ord("p"):
        cv2.waitKey(-1)
    if cv2.waitKey(90) & 0xFF == ord("q"):
        exit()
cv2.destroyAllWindows()
