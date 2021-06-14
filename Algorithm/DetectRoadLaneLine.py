import matplotlib.pylab as plt
import cv2
import numpy as np


image = cv2.imread("/home/tan/Python Source/video/test_img.png")  # Image test
imshape = image.shape
vertices = np.array(
    [
        [
            # (imshape[1] * 0.111, imshape[0]),
            (0, imshape[0]),
            (imshape[1] * 0.47, imshape[0] * 0.6),
            (imshape[1] * 0.53, imshape[0] * 0.6),
            (imshape[1], imshape[0]),
            # (imshape[1] * 0.889, imshape[0]),
        ]
    ],
    dtype=np.int32,
)

# vertices = np.array(
#     [
#         [
#             ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),
#             ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
#             (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
#             (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),
#         ]
#     ],
#     dtype=np.int32,
# )


def defind_vertices(image):
    return np.array(
        [
            [
                # (imshape[1] * 0.111, imshape[0]),
                (0, image.shape[0]),
                (image.shape[1] * 0.47, image.shape[0] * 0.6),
                (image.shape[1] * 0.53, image.shape[0] * 0.6),
                (image.shape[1], image.shape[0]),
                # (imshape[1] * 0.889, imshape[0]),
            ]
        ],
        dtype=np.int32,
    )


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


# = cv2.imread('road.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):
    print(image.shape)
    height = image.shape[0]
    # width = image.shape[1]
    # region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    # region_of_interest_vertices = [(400, height), (700, 1), (950, 1), (1200, height)]
    imshape = image.shape
    vertices = np.array(
        [
            [
                (imshape[1] * 0.223, imshape[0]),
                (imshape[1] * 0.446, imshape[0] * 0.75),
                (imshape[1] * 0.53, imshape[0] * 0.75),
                (imshape[1] * 0.777, imshape[0]),
            ]
        ],
        dtype=np.int32,
    )
    vertices2 = defind_vertices(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image, vertices2)
    # cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(
        cropped_image, rho=2, theta=np.pi / 180, threshold=15, lines=np.array([]), minLineLength=50, maxLineGap=20
    )

    ### New line ###
    if lines is None:
        print("Lines is None")
        return cropped_image
    # lines = cv2.HoughLinesP(
    #     cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100
    # )
    # print("line = ", lines)
    image_with_lines = drow_the_lines(image, lines)
    # return cropped_image
    return image_with_lines


def process2(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    # region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    # region_of_interest_vertices = [(400, height), (700, 1), (950, 1), (1200, height)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image, vertices)
    # cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(
        cropped_image, rho=1, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=50, maxLineGap=100
    )

    if lines is None:
        return cropped_image, canny_image
    # lines = cv2.HoughLinesP(
    #     cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100
    # )
    # print("line = ", lines)
    image_with_lines = drow_the_lines(image, lines)
    return cropped_image, image_with_lines


image = cv2.imread("/home/tan/Python Source/video/test_img3.png")
print(image.shape)
height = image.shape[0]
width = image.shape[1]
region_of_interest_vertices = [(400, height), (800, 730), (950, 730), (1200, height)]
# region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 120)
cropped_image = region_of_interest(
    canny_image,
    np.array([region_of_interest_vertices], np.int32),
)
lines = cv2.HoughLinesP(
    cropped_image, rho=1, theta=np.pi / 180, threshold=70, lines=np.array([]), minLineLength=50, maxLineGap=60
)
image_with_lines = drow_the_lines(image, lines)

### CODE ###
cropped_image, image_with_lines = process2(image)
list = [cropped_image, image_with_lines]
# plt.imshow()
fig = plt.figure(figsize=(image.shape[0], image.shape[1]))
for i in range(len(list)):
    fig.add_subplot(1, 2, i + 1)
    plt.imshow(list[i])
plt.show()

# cropped_image = region_of_interest(image, vertices)
# plt.imshow(cropped_image)
# plt.show()


# cap = cv2.VideoCapture("/home/tan/Python Source/video/test.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     frame = process(frame)
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# while cap.isOpened():
#     ret, frame = cap.read()
#     # frame = process(frame)
#     imshape = frame.shape
#     vertices = np.array(
#         [
#             [
#                 (imshape[1] * 0.111, imshape[0]),
#                 (imshape[1] * 0.47, imshape[0] * 0.7),
#                 (imshape[1] * 0.53, imshape[0] * 0.7),
#                 (imshape[1] * 0.889, imshape[0]),
#             ]
#         ],
#         dtype=np.int32,
#     )
#     frame = region_of_interest(frame, vertices)
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
