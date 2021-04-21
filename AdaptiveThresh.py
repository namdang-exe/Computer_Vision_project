import numpy as np
import cv2
import matplotlib.pyplot as plt


def display_lines(image, lines, x_start=None):
    if x_start is None:
        x_start = 0
    line_image = np.zeros_like(image)
    counter = 0
    x0 = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            if abs(y1 - y2) > abs(x1 - x2):
                cv2.line(line_image, (x1 + x_start, y1), (x2 + x_start, y2), (0, 0, 255), 3)
                if abs(x1 - x0) > 200:
                    counter += 1
                    x0 = x1
            print(counter)
    return line_image


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    low_threshold = 96
    high_threshold = 150
    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny


def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(100, height),  (150, 0),(430, 0), (600, height)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


img = cv2.imread('frame.jpg')
lane_image = img.copy()
canny_img = canny(lane_image)
cv2.imshow("canny_img", canny_img)
roi_img = roi(canny_img)
cv2.imshow("roi_img", roi_img)

lines = cv2.HoughLinesP(canny_img, 2, np.pi / 180, 50, minLineLength=10, maxLineGap=100)
line_image = display_lines(lane_image, lines)
cv2.imshow('line_image', line_image)
final_image = cv2.addWeighted(img, 1, line_image, 1, 1)
cv2.imshow('final_image', final_image)

# threshold = 100
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# cv2.imshow('thresh', thresh)

# _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
# dilated = cv2.dilate(thresh, None, iterations=1)
# cv2.imshow('thresh', thresh)

# img2 = img.copy()
# index = -1
# color = (0, 255, 255)

# _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     area = cv2.contourArea(contour)
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 3)
# if area > 1200:
#     cv2.drawContours(img2, contours, index, color, 3)

# cv2.imshow('Contours', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
