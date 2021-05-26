import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# Image processing
'''
im = cv2.imread('/home/tan/Python Source/Img/running.jpg')
thresh = canny(im)
cv2.imshow('result', thresh)
cv2.waitKey(0)
'''

# Video processing
videoCap = cv2.VideoCapture('/home/tan/Python Source/video/test2.mp4')
while (videoCap.isOpened()):
    _, frame = videoCap.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.Canny(imgray, 127, 255)  # nhị phân hóa ảnh
    cv2.imshow('result', thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
videoCap.release()
cv2.destroyAllWindows()
