import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

is_none_lines = False


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)

    upper_blue = np.array([179, 255, 255], dtype="uint8")
    lower_blue = np.array([98, 45, 0], dtype="uint8")

    # Morphological Transformations,Opening and Closing
    thresh = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 50, 100)
    cv2.imshow("edges", edges)

    return edges


def canny_edges(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 120)

    return edges


def region_of_interest(edges):
    height, width = edges.shape
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(edges)

    # only focus lower half of the screen
    polygon = np.array(
        [
            [
                (0, height),
                (0, height * 0.6),
                (width, height * 0.6),
                (width, height),
            ]
        ],
        np.int32,
    )

    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygon, 255)

    # A bitwise and operation between the mask and frame keeps only the rectangle area of the frame
    cropped_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow("roi", cropped_edges)

    return cropped_edges


def detect_lines(cropped_edges):
    rho = 1
    theta = np.pi / 180
    threshold = 10

    lines = cv2.HoughLinesP(cropped_edges, rho, theta, threshold, np.array([]), minLineLength=5, maxLineGap=150)

    return lines


def average_slope_intercept(frame, lines):
    lane_lines = []

    if lines is None:
        # print("no line segments detected")
        return lane_lines

    height, width, _ = frame.shape
    left_lines = []
    right_lines = []

    # Define the boundary for left and right line
    left_boundary = width * 0.6
    right_boundary = width * 0.4

    for i in range(0, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        # If it is not a line then skip
        if x1 == x2:
            continue

        fit = np.polyfit((x1, x2), (y1, y2), 1)
        # Calculate the slope and intercept of the line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)

        # Determine whether the line is left or right line
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            if x1 < left_boundary and x2 < left_boundary:
                left_lines.append((slope, intercept))
        else:
            if x1 > right_boundary and x2 > right_boundary:
                right_lines.append((slope, intercept))

    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines, then create left and right line
    left_lines_average = np.average(left_lines, axis=0)
    if len(left_lines) > 0:
        lane_lines.append(create_line(frame, left_lines_average))

    right_lines_average = np.average(right_lines, axis=0)
    if len(right_lines) > 0:
        lane_lines.append(create_line(frame, right_lines_average))

    return lane_lines


def create_line(frame, line):
    height, width, _ = frame.shape

    mean_slope, mean_intercept = line

    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if mean_slope == 0:
        mean_slope = 0.1

    # Sets start x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - mean_intercept) / mean_slope)
    # Sets end x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - mean_intercept) / mean_slope)

    return [[x1, y1, x2, y2]]


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_image = np.zeros_like(frame)

    # If list line is not None
    if lines is not None:
        for i in range(0, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            cv2.line(lines_image, (x1, y1), (x2, y2), line_color, line_width)

    lines_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)

    # Calculate steering angle of car
    steering_angle, num_of_lines = get_steering_angle(frame, lines)

    return lines_image, steering_angle, num_of_lines


def get_steering_angle(frame, lane_lines):

    height, width, _ = frame.shape
    num_of_lines = -1

    # Case 2 lines detected
    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)
        num_of_lines = 2

    # Case a line detected
    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)
        num_of_lines = 1

    # Case have no line detected
    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)
        num_of_lines = 0

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_degree = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_degree + 90

    return steering_angle, num_of_lines


def display_virtual_lane(frame, steering_angle, num_of_lines, line_color=(0, 0, 255), line_width=5):
    check_line = is_line_exist(num_of_lines)

    virtual_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi

    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(virtual_image, (x1, y1), (x2, y2), line_color, line_width)
    virtual_image = cv2.addWeighted(frame, 0.8, virtual_image, 1, 1)

    return virtual_image, check_line


def is_line_exist(num_of_lines):
    global is_none_lines

    # In case of not detecting any straight line
    if num_of_lines == 0:
        is_none_lines = True
        return is_none_lines

    # In case of detecting one or two lines
    else:
        # The previous situation was that no straight lines were detected
        if is_none_lines:
            is_none_lines = False
            return is_none_lines
        else:  # The previous situation was that have straight lines were detected
            return is_none_lines


def process(frame):
    canny_edge = detect_edges(frame)
    # canny_edge = canny_edges(frame)
    roi = region_of_interest(canny_edge)
    lines = detect_lines(roi)
    lane_lines = average_slope_intercept(frame, lines)
    lane_lines_image, steering_angle, num_of_lines = display_lines(frame, lane_lines)
    result_image, is_line_exist = display_virtual_lane(lane_lines_image, steering_angle, num_of_lines)

    return result_image, is_line_exist


def detect_lane(video, image, is_video):
    # Detect line handling
    global width

    if is_video:
        count = 0
        while video.isOpened():
            ret, frame = video.read()
            # width = frame.shape[1]
            frame, _ = process(frame)

            cv2.imshow("frame", frame)

            if cv2.waitKey(20) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()

    else:
        frame, _ = process(image)
        plt.imshow(frame)
        plt.show()


if __name__ == "__main__":
    # Capture video
    cap = cv2.VideoCapture("/home/tan/Python Source/video/test13.mp4")
    image = cv2.imread("/home/tan/Python Source/video/test13_img.png")  # Image test

    detect_lane(cap, image, is_video=True)
