import math

import cv2
import numpy as np
import sys
import time
import utils
import matplotlib.path as mplPath
import matplotlib.pyplot as plt


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def roi(image, roi_corners):
    mask = np.zeros_like(image)
    for roi_corner in roi_corners:
        polygons = np.array(roi_corner, np.int32)
        polygons = polygons.reshape((-1, 1, 2))
        cv2.polylines(mask, [polygons], True, (255, 255, 255), 3)
        cv2.fillPoly(mask, [polygons], (255, 255, 255))
        masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def filter_mask(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)
    return dilation


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = height
    y2 = int(y1 * (2 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def draw_line(image, lines, color=True):
    # line_image = np.zeros_like(image)
    line_image = image.copy()
    color_value = (0, 255, 255)
    if not color:
        color_value = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
            cv2.line(line_image, (x1, y1), (x2, y2), color_value, 2)
    return line_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    height = image.shape[0]
    width = image.shape[1]
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            dx = x1 - x2
            dy = y1 - y2
            # remove lines without slanted slopes
            if abs(abs(dx) - abs(dy)) > 30:
                continue
            # remove horizontal lines
            elif abs(dy) < 10:
                continue
            else:
                [vx, vy, x0, y0] = cv2.fitLine(np.array([[x1, y1], [x2, y2]]), cv2.DIST_L2, 0, 0.01, 0.01)
                # m == length
                m = 50
                cv2.line(line_image, (x0 - m * vx[0], y0 - m * vy[0]), (x0 + m * vx[0], y0 + m * vy[0]), (0, 0, 255), 1)
                cv2.line(roiImg, (x0 - m * vx[0], y0 - m * vy[0]), (x0 + m * vx[0], y0 + m * vy[0]), (0, 0, 255), 1)
                # separate lines
                x_start = x0 - m * vx[0]
                y_start = y0 - m * vy[0]
                x_end = x0 + m * vx[0]
                y_end = y0 + m * vy[0]
                if (x_start - x_end) == 0:
                    continue
                m1 = slope(x_start, y_start, x_end, y_end)
                b1 = y_start - m1 * x_start
                # middle line that check from left to right
                divide_starty, divide_endy = 195, 195
                m0 = 0  # slope of a horizontal line = 0
                b0 = divide_starty
                # check intersection
                if (m0 - m1) != 0 and not math.isnan(m0 - m1):
                    x_intercept = (b1 - b0) / (m0 - m1)
                    x_percentage = (x_intercept / width) * 100

                    if 15 <= x_percentage <= 30:
                        cv2.putText(line_image, '1', (115, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
                        left_lane1.append((m1, b1))
                    elif 35 <= x_percentage <= 45:
                        cv2.putText(line_image, '2', (157, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
                        right_lane1.append((m1, b1))
                        left_lane2.append((m1, b1))
                    elif 55 <= x_percentage <= 68:
                        cv2.putText(line_image, '3', (197, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
                        right_lane2.append((m1, b1))
                        left_lane3.append((m1, b1))
                    elif 72 <= x_percentage <= 90:
                        cv2.putText(line_image, '4', (237, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
                        right_lane3.append((m1, b1))
                    else:
                        continue
    return line_image


def average_slope_intercept(image, left_lane_parameter, right_lane_parameter):
    line_image = image.copy()
    # find average slope m and intercept b         |         y = mx + b
    left_lane_average = np.average(left_lane_parameter, axis=0)
    right_lane_average = np.average(right_lane_parameter, axis=0)
    # turn into x1,y1,x2,y2
    left_line = make_coordinates(line_image, left_lane_average)
    right_line = make_coordinates(line_image, right_lane_average)
    lane_lines = np.array([left_line, right_line])
    return lane_lines


def draw_lane_lines(image, left_lane_parameter, right_lane_parameter):
    line_image = image.copy()
    # find average slope m and intercept b         |         y = mx + b
    left_lane_average = np.average(left_lane_parameter, axis=0)
    right_lane_average = np.average(right_lane_parameter, axis=0)
    # turn into x1,y1,x2,y2
    left_line = make_coordinates(line_image, left_lane_average)
    right_line = make_coordinates(line_image, right_lane_average)
    lane_lines = np.array([left_line, right_line])
    line_image = draw_line(line_image, lane_lines)
    return line_image


def find_lines(image, num_lines, time):
    pass


def remove_duplicates(line_parameter):
    return [i for n, i in enumerate(line_parameter) if i not in line_parameter[:n]]


def on_segment(p, q, r):
    '''Given three colinear points p, q, r, the function checks if
    point q lies on line segment "pr"
    '''
    if (max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and
            max(p[1], r[1]) >= q[1] >= min(p[1], r[1])):
        return True
    return False


def orientation(p, q, r):
    '''Find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    '''

    val = ((q[1] - p[1]) * (r[0] - q[0]) -
           (q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0  # collinear
    elif val > 0:
        return 1  # clockwise
    else:
        return 2  # counter-clockwise


def do_intersect(p1, q1, p2, q2):
    """Main function to check whether the closed line segments p1 - q1 and p2
       - q2 intersect"""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False  # Doesn't fall in any of the above cases


def make_tan_line(img, pt1, pt2, box):
    tan_image = img.copy()
    tan_line = None
    if (pt1[0] - pt2[0]) != 0:
        if (pt1[1] - pt2[1]) / (pt1[0] - pt2[0]) != 0:
            # draw a tan line on the center of the car
            mid_pt = ((pt2[0] + pt1[0]) // 2, (pt2[1] + pt1[1]) // 2)
            m = slope(pt1[0], pt1[1], pt2[0], pt2[1])
            m = -1 / m
            b = mid_pt[1] - m * mid_pt[0]
            # the end and start point of the tan line = the intercept of the line and the bounding box
            effs1 = np.polyfit((box[1][0], box[2][0]), (box[1][1], box[2][1]), 1)  ## m, b
            poly1 = np.poly1d(effs1)
            x1 = int((b - effs1[1]) / (effs1[0] - m))
            y1 = int(poly1(x1))
            effs2 = np.polyfit((box[3][0], box[0][0]), (box[3][1], box[0][1]), 1)  ## m, b
            poly2 = np.poly1d(effs2)
            x2 = int((b - effs2[1]) / (effs2[0] - m))
            y2 = int(poly2(x2))
            cv2.line(tan_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            tan_line = np.array([x1, y1, x2, y2])
    return tan_line


def find_equal_line(roi_corners, y):
    left_line_parameter = np.polyfit((roi_corners[0][0], roi_corners[1][0]), (roi_corners[0][1], roi_corners[1][1]), 1)
    right_line_parameter = np.polyfit((roi_corners[2][0], roi_corners[3][0]), (roi_corners[2][1], roi_corners[3][1]), 1)
    m1, b1 = left_line_parameter
    m2, b2 = right_line_parameter
    # finds the intercept of two lanes
    x0 = int((b2 - b1) / (m1 - m2))
    y0 = int(m1 * x0 + b1)
    # point on the left lane
    x = int((y - b1) / m1)
    # performs bin search
    uV = np.array([(x0 - x), (y0 - y)])
    high = y0
    low = max(roi_corners[0][1], roi_corners[1][1], roi_corners[2][1], roi_corners[3][1])
    while abs(high - low) > 1:
        mid_y = (high + low) // 2
        mid_x = int((mid_y - b2) / m2)
        vV = np.array([(x0 - mid_x), (y0 - mid_y)])
        if abs(np.linalg.norm(uV) - np.linalg.norm(vV)) < 0.01:
            break
        elif np.linalg.norm(uV) > np.linalg.norm(vV):
            high = mid_y
        else:
            low = mid_y

    return np.array([x, y, mid_x, mid_y])


def find_intercept(first_line_parameter, second_line_parameter):
    m0, b0 = first_line_parameter
    m1, b1 = second_line_parameter
    x = int((b0 - b1) / (m1 - m0))
    y = int(m0 * x + b0)
    intercept = np.array([x, y])
    return intercept


def calc_proportion(car_left_extreme, car_right_extreme, left_lane_parameter, right_lane_parameter):
    """
    This function will calculate the proportion between left, right extreme of the car and two sides of the lane
    We achieve this by compute the length from the left to the right lane (this line segment has the same vector as car extremes's line segment)
    Let's call it 'total'
    We then calculate the length from the left extreme to the left lane and right extreme to the right lane
    Call it 'left' and 'right'
    :param car_left_extreme:
    :param car_right_extreme:
    :param left_lane_parameter:
    :param right_lane_parameter:
    :return: left proportion and right proportion
    """
    # find slope of two extremes
    m, b = np.polyfit((car_left_extreme[0], car_right_extreme[0]), (car_left_extreme[1], car_right_extreme[1]), 1)

    # With the same vector as car's extremes, find 2 points on the left and right side of the lane
    left_lane_pts = find_intercept((m, b), left_lane_parameter)
    right_lane_pts = find_intercept((m, b), right_lane_parameter)
    total_vector = np.array([(left_lane_pts[0] - right_lane_pts[0]), (left_lane_pts[1] - right_lane_pts[1])])
    total = np.linalg.norm(total_vector)
    left_vector = np.array([(car_left_extreme[0] - left_lane_pts[0]), (car_left_extreme[1] - left_lane_pts[1])])
    left = np.linalg.norm(left_vector)
    right_vector = np.array([(car_right_extreme[0] - right_lane_pts[0]), (car_right_extreme[1] - right_lane_pts[1])])
    right = np.linalg.norm(right_vector)
    left_proportion = (left / total) * 100
    right_proportion = (right / total) * 100
    return np.array([left_proportion, right_proportion])


def draw_roi_track(img, roi_corners, contour, region=False, lines=None, lanes=None):
    """
    Car Tracking Function
    :param img:
    :param roi_corners:
    :param box:
    :param region:
    :param lines: lines == number of lines on the region
    :return:
    """
    global counter
    global left_lane
    global right_lane
    mask = img.copy()

    # bounding box contour points
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), theta = rect
    box = cv2.boxPoints(((x, y), (w, h), theta))
    box = np.int0(box)

    # mid_pt1, mid_pt2 = upper and lower mid point of the car
    mid_pt1 = ((box[2][0] + box[3][0]) // 2, (box[2][1] + box[3][1]) // 2)
    mid_pt2 = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)

    # finish line
    x0, y0 = roi_corners[0][0], roi_corners[0][1]
    x1, y1 = roi_corners[3][0], roi_corners[3][1]

    # ensures the car is inside the roi corner
    path = mplPath.Path(roi_corners)
    # reshape the list of contour points to [[x,y],[x1,y1],...]
    points = contour.reshape(-1, 2)
    X = points[:, 0]
    Y = points[:, 1]
    n_points = list(zip(X, Y))
    if path.contains_points(n_points).all():
        print(abs(theta))
        cv2.putText(mask, str(abs(theta)), (box[2][0], box[2][1] - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        theta_avg.append(theta)
        # displays left and right extremes on mask
        cv2.circle(mask3, (box[1][0], box[1][1]), 3, (0, 255, 0), -1)
        cv2.circle(mask3, (box[0][0], box[0][1]), 3, (0, 0, 255), -1)
        # displays left and right extremes on car
        cv2.circle(mask, (box[1][0], box[1][1]), 10, (0, 255, 0), -1)
        cv2.circle(mask, (box[0][0], box[0][1]), 10, (0, 0, 255), -1)
        cv2.line(mask1, (box[1][0], box[1][1]), (box[0][0], box[0][1]), (0, 0, 255))
        # this can be improved
        # appends left and right extremes to the array
        left_pts = np.array([[box[1][0], box[1][1]]])
        left_lane = np.concatenate((left_lane, left_pts), axis=0)
        right_pts = np.array([[box[0][0], box[0][1]]])
        right_lane = np.concatenate((right_lane, right_pts), axis=0)
        # draw bounding box
        cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)

        if region:
            # draw regions
            boxLength = abs(roi_corners[1][1] - roi_corners[0][1])
            num_regions = lines + 1
            eps = boxLength - ((boxLength // num_regions) * num_regions)
            # splits into multiple region
            for step in np.arange(boxLength // num_regions + eps, boxLength, boxLength // num_regions):
                region_pts = find_equal_line(roi_corners, roi_corners[1][1] + step - 20)
                x_left, y_left, x_right, y_right = region_pts.reshape(4)

                # lines to separate multiple regions
                effs = np.polyfit((x_left, x_right), (y_left, y_right), 1)
                poly = np.poly1d(effs)
                # when car intercepts region lines, then report
                if do_intersect(mid_pt1, mid_pt2, (x_left, y_left), (x_right, y_right)):
                    cv2.line(mask, (x_left, y_left), (x_right, y_right), (0, 0, 255), 3)

        else:
            effs = np.polyfit((x0, x1), (y0, y1), 1)
            poly = np.poly1d(effs)
            if abs(mid_pt2[1] - poly(mid_pt2[0])) <= 10:
                cv2.line(mask, (x0, y0), (x1, y1), (0, 0, 255), 4)
                counter += 1
    return mask


def draw_new_lines(img, y_min, y_max, coef_left, coef_right, angle=None, lines=30, left_percent=None,
                   right_percent=None):
    if angle is None:
        angle = 0
    else:
        angle = abs(angle)
    minH = y_min
    maxH = y_max
    range = maxH - minH
    m0, b0 = coef_left
    m1, b1 = coef_right
    mask = img.copy()
    if len(img.shape) != 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if left_percent is None:
        left_percent = 26.786
    if right_percent is None:
        right_percent = 14.723
    if angle == 0:
        counter = 0
        for y0 in np.arange(minH, maxH, range // lines):
            x0 = int((y0 - b0) / m0)
            y1 = y0
            x1 = int((y1 - b1) / m1)
            # cv2.line(mask, (x0, y0), (x1, y1), (0, 0, 255), 2)

            # calculate new lines
            total_vector = np.array([(x0 - x1), (y0 - y1)])
            total = np.linalg.norm(total_vector)
            left_percentage = float(left_percent / 100)
            right_percentage = float(right_percent / 100)
            # this can be a problem for curve lanes
            x0 = x0 - int(total * left_percentage)
            x1 = x1 + int(total * right_percentage)
            if counter < lines // 4:
                x0 = x0 + 1
            elif counter < lines // 2:
                x0 = x0 - 2
                x1 = x1 + 2
            elif counter < 3 * lines // 4:
                x0 = x0 - 3
                x1 = x1 + 3
            else:
                x0 = x0 - 5
                x1 = x1 + 3
            cv2.line(mask, (x0, y0), (x1, y1), (0, 255, 0), 2)
            counter += 1

    elif angle != 0:
        y0 = minH
        y1 = maxH
        for x0 in np.arange(minW, minW + 60, minW // lines):
            dX = int(maxH / np.tan(np.radians(angle)))
            x1 = x0 - dX
            cv2.line(mask, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return mask


cap = cv2.VideoCapture("car4.mp4")

_, frame = cap.read()
W = frame.shape[1]
H = frame.shape[0]
frame = cv2.resize(frame, (W // 2, H // 2))
first_frame = frame.copy()
height = frame.shape[0]
width = frame.shape[1]

# blank image
blank_image = np.ones_like(frame)
blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
blank_image *= 255
mask1 = np.zeros_like(frame)
lane_image = frame.copy()
mask3 = np.zeros_like(frame)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Region of Interest
roi_corners = [
    [[0, 305], [205, 180], [296, 180], [250, height - 10]],
    [[395, height - 10], [340, 180], [426, 180], [width, 315]]
]
# roi_corners1 = [[396, 351], [340, 180], [370, 180], [490, 339]]
# roi_corners1 = [[376, 351], [320, 180], [390, 180], [510, 339]]
roi_corners1 = [[122, 342],
                [248, 180],
                [317, 180],
                [269, 352]]

left_lane = np.array([[0, 0]])
right_lane = np.array([[0, 0]])
frame_counter = 0
point_counter = 0
fps = 0
counter = 0
theta_avg = []
left_avg = []
right_avg = []
i = 0
while True:
    if frame_counter == 0:
        time_start = time.time()
    frame_counter += 1  # measure fps
    ret, frame = cap.read()

    if not ret:
        print("***ERROR*** Can't read frame")
        break

    frame = cv2.resize(frame, (W // 2, H // 2))
    mask2 = frame.copy()
    roiImg = roi(frame, roi_corners)

    line_image = roiImg.copy()
    # lane_image = frame.copy()
    # roi_blur = cv2.GaussianBlur(roiImg, (5, 5), 1)
    roi_blur = cv2.blur(roiImg, (5, 5))

    # remove background
    fgMask = fgbg.apply(roi_blur)
    # filter background
    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('org', fgMask)
    # fgMask = cv2.bitwise_or(fgMask, fgMask, mask=blank_image)
    fgMask = filter_mask(fgMask)
    # show results on different frames
    cntImg = fgMask.copy()
    cntImg = cv2.cvtColor(cntImg, cv2.COLOR_GRAY2BGR)
    cntImg2 = fgMask.copy()
    cntImg2 = cv2.cvtColor(cntImg2, cv2.COLOR_GRAY2BGR)
    trackImg = fgMask.copy()
    trackImg = cv2.cvtColor(trackImg, cv2.COLOR_GRAY2BGR)

    # --------------------- Find lane lines ------------------#
    road = fgbg.getBackgroundImage()
    line1 = [1, 307, 206, 180]
    line2 = [34, 351, 233, 180]
    #################
    line3 = [142, 342, 268, 180]
    line4 = [249, 352, 297, 180]
    #################
    line5 = [396, 351, 340, 180]
    line6 = [490, 339, 370, 180]
    #################
    line7 = [572, 328, 397, 180]
    line8 = [640, 315, 426, 180]
    lines = np.array([line1, line2, line3, line4, line5, line6, line7, line8])
    lines2 = np.array([line3, line4])
    # left_line_parameter = np.polyfit((line5[0], line5[2]), (line5[1], line5[3]), 1)
    # right_line_parameter = np.polyfit((line6[0], line6[2]), (line6[1], line6[3]), 1)

    # ------------------------- End Find Lane Lines ---------------------- #

    # ------------------------- Draw Lane Lines -------------------------- #
    # displays roi
    roi_pts = np.array(roi_corners1, np.int32)
    roi_pts = roi_pts.reshape((-1, 1, 2))
    cv2.polylines(trackImg, [roi_pts], True, (255, 255, 255), 3)

    trackImg = draw_line(trackImg, lines2)
    cntImg = draw_line(cntImg, lines)
    cntImg2 = draw_line(cntImg2, lines)
    mask1 = draw_line(mask1, lines2)
    mask3 = draw_line(mask3, lines2)
    # -------------------------- End Draw Lane Lines --------------------- #

    # ---------------------- Draw Contours ------------------------- #
    # find contours
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    center = None

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= 35) and (
                h >= 35) and (w <= 100) and (h <= 100)
        if not contour_valid:
            continue
        else:
            epsilon = 0.000001 * cv2.arcLength(contour, False)
            contour = cv2.approxPolyDP(contour, epsilon, False)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if center is not None:
                # draw contours points
                cv2.drawContours(cntImg2, contours, -1, (0, 0, 255))

                # draw rotated rectangles
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                (x, y), (w, h), theta = rect
                box = cv2.boxPoints(((x, y), (w, h), theta))
                box = np.int0(box)
                cv2.drawContours(cntImg, [box], 0, (0, 0, 255), 2)

                # filter out bad rectangles
                if not math.isnan(angle) and angle != 0 and 40 <= abs(angle) <= 80:
                    trackImg = draw_roi_track(trackImg, roi_corners1, contour, False, 2)

    # left lane
    if i == 0:
        left_lane = left_lane[1:]
        right_lane = right_lane[1:]
        i += 1

    if not left_lane.any():
        continue
    else:
        # left side
        coef = np.polyfit(left_lane[:, 0].reshape(-1, 1).ravel(), left_lane[:, 1].reshape(-1, 1).ravel(), 1)
        left_x = left_lane[:, 0].reshape(-1, 1).ravel()
        poly = np.poly1d(coef)
        new_x = np.linspace(min(left_x), max(left_x))  # can be improved
        new_x = np.int0(new_x)
        new_y = np.int0(poly(new_x))
        mat = list(zip(new_x, new_y))
        mat = np.array(mat)
        cv2.polylines(mask2, [mat], False, (0, 0, 0), 3)
        # right side
        right_x = right_lane[:, 0].reshape(-1, 1).ravel()
        coef = np.polyfit(right_x, right_lane[:, 1].reshape(-1, 1).ravel(), 1)
        poly = np.poly1d(coef)
        new_x = np.linspace(min(right_x), max(right_x))  # can be improved
        new_x = np.int0(new_x)
        new_y = np.int0(poly(new_x))
        mat = list(zip(new_x, new_y))
        mat = np.array(mat)
        cv2.polylines(mask2, [mat], False, (0, 0, 0), 3)

    cv2.imshow('cntImg', cntImg)
    cv2.imshow('cntImg2', cntImg2)
    cv2.imshow('frame', frame)
    cv2.imshow('trackImg', trackImg)
    cv2.imshow('mask1', mask1)
    cv2.imshow('mask2', mask2)
    cv2.imshow('mask3', mask3)

    # get time elapsed until 1 second, compute FPS
    time_elapsed = time.time() - time_start
    if time_elapsed >= 1.0:
        fps = frame_counter
        frame_counter = 0
        print("FPS: " + str(fps))

    k = cv2.waitKey(1) & 0xff

    if k == ord('p'):
        k = cv2.waitKey(0)
    if k == 13:
        cv2.imwrite("lane_preprocess.png", blank_image)
    if k == 27:
        break
    if k == ord('q'):
        break

    counter += 1

print(left_lane)
print('theta avg', np.mean(np.array(theta_avg)))
from scipy import stats

sum = np.sum(left_lane, axis=1)
print(sum)
z = np.abs(stats.zscore(sum))
print(z)
sum = sum[(z < 3)]
print(sum)
pts = left_lane[np.where(z < 3)]
print(pts)

# ------------------------ Draw lane ---------------------------- #

# left lane
# separated by x-coordinate only
left_x = left_lane[:, 0].reshape(-1, 1).ravel()
# left_x = left_x - 16
# separated by y-coordinate only
left_y = left_lane[:, 1].reshape(-1, 1).ravel()
plt.hist2d(left_x, left_y)
plt.show()
# left_y = left_y - 30
coef_left = np.polyfit(left_x, left_y, 1)

poly = np.poly1d(coef_left)
new_left_x = np.linspace(min(left_x), max(left_x))  # can be changed according to y later
new_left_x = np.int0(new_left_x)
new_left_y = np.int0(poly(new_left_x))
left = list(zip(new_left_x, new_left_y))
left = np.array(left)
final = cv2.polylines(first_frame, [left], False, (0, 0, 0), 3)
# right side
right_x = right_lane[:, 0].reshape(-1, 1).ravel()
right_x = right_x + 9
right_y = right_lane[:, 1].reshape(-1, 1).ravel()
right_y = right_y + 5
coef_right = np.polyfit(right_x, right_y, 1)
poly = np.poly1d(coef_right)
new_right_x = np.linspace(min(right_x), max(right_x))
new_right_x = np.int0(new_right_x)
new_right_y = np.int0(poly(new_right_x))
right = list(zip(new_right_x, new_right_y))
right = np.array(right)
final = cv2.polylines(first_frame, [right], False, (0, 0, 0), 3)
final = draw_new_lines(first_frame, 225, height, coef_left, coef_right)

cv2.imshow('result', final)

final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
plt.imshow(final)
plt.show()

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
