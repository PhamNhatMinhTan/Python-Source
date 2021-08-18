import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import roi


width = 0
is_none_lines = False


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_blue = np.array([90, 120, 0], dtype="uint8")
    # upper_blue = np.array([150, 255, 255], dtype="uint8")
    upper_blue = np.array([130, 255, 255], dtype="uint8")
    lower_blue = np.array([90, 120, 0], dtype="uint8")
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # detect edges
    edges = cv2.Canny(mask, 50, 100)

    return edges


def canny_edges(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 120)

    return edges


def region_of_interest(edges):
    # height, width = edges.shape
    height = edges.shape[0]
    width = edges.shape[1]
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(edges)

    # only focus lower half of the screen
    polygon = np.array(
        [
            [
                (0, height),
                (0, height / 2),
                (width, height / 2),
                (width, height),
            ]
        ],
        np.int32,
    )

    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygon, 255)

    # A bitwise and operation between the mask and frame keeps only the rectangle area of the frame
    cropped_edges = cv2.bitwise_and(edges, mask)
    # cv2.imshow("roi", cropped_edges)

    return cropped_edges


def detect_lines(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 10

    lines = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, np.array([]), minLineLength=5, maxLineGap=150)

    return lines


def average_slope_intercept(frame, lines):
    lane_lines = []

    if lines is None:
        # print("no line segments detected")
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    # Define the boundary for left and right line
    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line in lines:
        for x1, y1, x2, y2 in line:
            # If it is not a line then skip
            if x1 == x2:
                # print("skipping vertical lines (slope = infinity")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            # Calculate the slope and intercept of the line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            # Determine whether the line is left or right line
            # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
                # print("Left Slope, intercept = {0}, {1}".format(slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
                # print("Right Slope, intercept = {0}, {1}".format(slope, intercept))

    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines, then create left and right line
    left_fit_average = np.average(left_fit, axis=0)
    # print("left_fit_average = " + str(left_fit_average))
    if len(left_fit) > 0:
        lane_lines.append(create_line(frame, left_fit_average))
        # print("LEFT LINE: " + str(create_line(frame, left_fit_average)))

    right_fit_average = np.average(right_fit, axis=0)
    # print("right_fit_average = " + str(right_fit_average))
    if len(right_fit) > 0:
        lane_lines.append(create_line(frame, right_fit_average))
        # print("RIGHT LINE: " + str(create_line(frame, right_fit_average)))

    return lane_lines


def create_line(frame, line):
    height, width, _ = frame.shape

    slope, intercept = line

    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0:
        slope = 0.1

    # Sets start x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets end x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_image = np.zeros_like(frame)

    # If list line is not None
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
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
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    # TEST #
    # print("\n")
    # print("x - y offset: {0}, {1}".format(x_offset, y_offset))
    # print("radian - deg - steer angle: {0}, {1}, {2}".format(angle_to_mid_radian, angle_to_mid_deg, steering_angle))
    # print("\n")

    return steering_angle, num_of_lines


def display_virtual_lane(frame, steering_angle, num_of_lines, line_color=(0, 0, 255), line_width=5):
    check_line = is_line_exist(num_of_lines)

    virtual_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi

    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    # x2 = int(x1 - height / 2)
    y2 = int(height / 2)

    # Test value
    # print("Tan of steering angle: " + str(math.tan(steering_angle_radian)))
    # print("Height/2: " + str(height / 2 / math.tan(steering_angle_radian)))
    # print("x1: " + str(x1))
    # print("x2: " + str(x2))

    cv2.line(virtual_image, (x1, y1), (x2, y2), line_color, line_width)
    virtual_image = cv2.addWeighted(frame, 0.8, virtual_image, 1, 1)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(virtual_image, signal, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return virtual_image, check_line


def is_line_exist(num_of_lines):
    global is_none_lines

    if num_of_lines == 0:
        # is_turn_right = True
        is_none_lines = True
        # return direction_no_line
        # return "TURN RIGHT"
        return is_none_lines

    elif num_of_lines == 1:
        is_none_lines = False
        return is_none_lines


def process(frame):
    canny_edge = detect_edges(frame)
    # canny_edge = canny_edges(frame)
    roi_l = region_of_interest(canny_edge)
    lines = detect_lines(roi_l)
    # # frame = roi.defind_region_boundary(frame, "both")
    lane_lines = average_slope_intercept(frame, lines)
    lane_lines_image, steering_angle, num_of_lines = display_lines(frame, lane_lines)
    # steering_angle, num_of_line = get_steering_angle(frame, lane_lines)
    # signal = handle_signal(steering_angle, lane_lines)
    result_image, is_line_exist = display_virtual_lane(lane_lines_image, steering_angle, num_of_lines)

    return result_image, is_line_exist


def detect_lane(video, image, is_video):
    # Detect line handling
    # global width
    width = int(video.get(3))
    height = int(video.get(4))

    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter("out_video.mp4", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (width, height))
    if is_video:
        while video.isOpened():
            ret, frame = video.read()
            # width = frame.shape[1]
            frame, _ = process(frame)
            # out.write(frame)
            cv2.imshow("frame", frame)

            if cv2.waitKey(90) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()

    else:
        frame, _ = process(image)
        plt.imshow(frame)
        plt.show()


def test_hsv(video, type):
    # Detect line handling
    while video.isOpened():
        ret, frame = video.read()
        if type == "hsv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow("frame", frame)

        if type == "canny":
            frame = detect_edges(frame)
            cv2.imshow("frame", frame)

        if type == "roi":
            # frame = canny_edges(frame)
            frame = detect_edges(frame)
            frame = region_of_interest(frame)
            cv2.imshow("frame", frame)

        if cv2.waitKey(40) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


def test_image(image, type):
    if type == "hsv":
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        plt.imshow(frame)

    if type == "canny":
        frame, _ = process(image)
        # frame = roi.defind_region_boundary(frame, "both")
        # cv2.imwrite("virtual-line.jpg", frame)
        plt.imshow(frame)

    if type == "roi":
        # frame = canny_edges(frame)
        # frame = detect_edges(image)
        frame = region_of_interest(image)
        plt.imshow(frame)

    plt.show()


if __name__ == "__main__":
    # Capture video
    cap = cv2.VideoCapture("/home/tan/Python Source/video/test11.mp4")
    # cap = cv2.VideoCapture(0)
    # image = cv2.imread("/home/tan/Python Source/video/test9_img.png")  # Image test
    image = cv2.imread("/home/tan/Python Source/video/real/img2_test.jpg")  # Image test
    # image = cv2.imread("/home/tan/Python Source/video/real/img_test3.png")  # Image test

    detect_lane(cap, image, is_video=True)
    # test_image(image, "canny")
    # test_hsv(cap, "canny")
