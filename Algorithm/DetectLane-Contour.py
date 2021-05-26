import numpy as np
import matplotlib.pyplot as plt
import cv2


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def defind_vertices(image):
    return np.array(
        [
            [
                # (imshape[1] * 0.111, imshape[0]),
                (0, image.shape[0]),
                (image.shape[1] * 0.47, image.shape[0] * 0.72),
                (image.shape[1] * 0.53, image.shape[0] * 0.72),
                (image.shape[1], image.shape[0]),
                # (imshape[1] * 0.889, imshape[0]),
            ]
        ],
        dtype=np.int32,
    )


def process(image):
    print(image.shape)
    original_image = image.copy()
    canny_image = canny(image)  # nhị phân hóa ảnh
    # crop ảnh
    vertices = defind_vertices(image)
    cropped_image = region_of_interest(canny_image, vertices)
    # thresh = cv2.Canny(imgray, 127, 255)  # nhị phân hóa ảnh
    contours, hierarchy = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # vẽ lại ảnh contour vào ảnh gốc
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)
    return original_image


cap = cv2.VideoCapture("/home/tan/Python Source/video/test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
