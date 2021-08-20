import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime


def test_plt():
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        img = np.random.randint(10, size=(h, w))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def test_npAverage():
    a = np.arange(6).reshape(3, 2)

    print("Our array is:")
    print(a)
    print("\n")

    print("Modified array:")
    wt = np.array([3, 5])
    print(wt)
    print("\n")

    print("Sum of Multiply 2 array")
    print(sum(a @ wt) / sum(wt))

    print(np.average(a, axis=1, weights=wt))
    print("\n")


def testHistogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = cv2.equalizeHist(hsv[:, :, 0])

    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(out)
    plt.show()


def show_hsv_equalized(image):
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
    return eq_image
    # plt.imshow(eq_image)
    # plt.show()


def test_current_time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    # hour = current_time.hour
    print("Current Time =", current_time)
    print("Hour = ", str(now.hour))
    print("type: " + str(type(now.hour)))


def display_lines(frame, line_color=(0, 255, 0), line_width=6):

    height, width, _ = frame.shape

    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down
    x1 = -500
    x2 = 1500

    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_image = np.zeros_like(frame)
    cv2.line(lines_image, (x1, y1), (x2, y2), line_color, line_width)

    lines_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)

    plt.imshow(lines_image)
    plt.show()

    return lines_image


if __name__ == "__main__":

    image = cv2.imread("/home/tan/Python Source/video/test13_img.png")  # Image test
    # image = cv2.imread("/home/tan/Python Source/video/real/img2_test.jpg")  # Image test
    # image = cv2.imread("/home/tan/Python Source/video/real/img_test3.png")  # Image test

    # testHistogram(image)
    # show_hsv_equalized(image)
    # test_current_time()
    # test_npAverage()
    display_lines(image)
