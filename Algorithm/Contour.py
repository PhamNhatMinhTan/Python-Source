import numpy as np
import cv2
import matplotlib.pyplot as plt


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# Image processing
"""
im = cv2.imread('/home/tan/Python Source/Img/sukudo.jpg')
# chuyển ảnh xám thành ảnh grayscale
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(imgray, 127, 255)  # nhị phân hóa ảnh
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# vẽ lại ảnh contour vào ảnh gốc
cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

# show ảnh lên
cv2.imshow("ballons", im)
cv2.waitKey(0)
"""

# Contour Video
videoCapture = cv2.VideoCapture("/home/tan/Python Source/video/test.mp4")
while videoCapture.isOpened():
    _, frame = videoCapture.read()
    thresh = canny(frame)  # nhị phân hóa ảnh
    # thresh = cv2.Canny(imgray, 127, 255)  # nhị phân hóa ảnh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # vẽ lại ảnh contour vào ảnh gốc
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
videoCapture.release()
cv2.destroyAllWindows()
