import cv2

cap = cv2.VideoCapture('/dev/video0')
ret, frame = cap.read()
print(ret, frame)
