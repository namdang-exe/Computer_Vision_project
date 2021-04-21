import cv2
import numpy as np
import math
import time
import pandas as pd
import matplotlib.path as mplPath

cap = cv2.VideoCapture('car4.mp4')
_, frame = cap.read()
W = frame.shape[1]
H = frame.shape[0]
frame = cv2.resize(frame, (W // 2, H // 2))
first_frame = frame.copy()
height = frame.shape[0]
width = frame.shape[1]
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

roi_corners = [
    [[0, 260], [162, 180], [296, 180], [250, height - 10]],
    [[395, height - 10], [340, 180], [470, 180], [width + 30, 315]]
]
# --------------------- Find lane lines ------------------ #

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
#################
lines = np.array([line2, line3, line4])
lines2 = np.array([line3, line4])
lines0 = np.array([line1, line2])
lines1 = np.array([line2, line3])

# left_line_parameter = np.polyfit((line5[0], line5[2]), (line5[1], line5[3]), 1)
# right_line_parameter = np.polyfit((line6[0], line6[2]), (line6[1], line6[3]), 1)

# ------------------------- End Find Lane Lines ---------------------- #
indices = []
data = []
frame_counter = 0
fps = 0
counter = 0
car_count = 0
start_pt = [[27, 287], [248, 360]]
end_pt = [[200, 181], [292, 202]]
num_region = 10


def roi(image, roi_corners):
    mask = np.zeros_like(image)
    for roi_corner in roi_corners:
        polygons = np.array(roi_corner, np.int32)
        polygons = polygons.reshape((-1, 1, 2))
        cv2.polylines(mask, [polygons], True, (255, 255, 255), 3)
        cv2.fillPoly(mask, [polygons], (255, 255, 255))
        masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def point_on_tan_line(point, tan_line):
    # solving for x,y coord of the point
    x0, y0 = point[0], point[1]
    x1, y1, x2, y2 = tan_line[:]
    # tan_line || y = mx + b
    if (x1 - x2) != 0:
        m, b = np.polyfit((x1, x2), (y1, y2), 1)
        A = np.array([[m, -1], [(x1 - x2), (y1 - y2)]])
        b = np.array([[-b], [(x1 - x2) * x0 + (y1 - y2) * y0]])
        z = np.linalg.solve(A, b)
        return int(z[0]), int(z[1])


def find_equal_line(start_pt, end_pt, y):
    # takes a point on the right lane and generates a point on the left lane
    left_line_param = np.polyfit((start_pt[0][0], end_pt[0][0]), (start_pt[0][1], end_pt[0][1]), 1)
    right_line_param = np.polyfit((start_pt[1][0], end_pt[1][0]), (start_pt[1][1], end_pt[1][1]), 1)
    m1, b1 = left_line_param
    m2, b2 = right_line_param
    # finds the intercept of two lanes
    x0 = int((b2 - b1) / (m1 - m2))
    y0 = int(m1 * x0 + b1)
    # starts with a point on the right lane
    x = int((y - b2) / m2)
    # performs binary search
    uV = np.array([(x0 - x), (y0 - y)])
    high = y0
    low = max(start_pt[0][1], start_pt[1][1], end_pt[0][1], end_pt[1][1])
    while abs(high - low) > 1:
        mid_y = (high + low) // 2
        mid_x = int((mid_y - b1) / m1)
        vV = np.array([(x0 - mid_x), (y0 - mid_y)])
        if abs(np.linalg.norm(uV) - np.linalg.norm(vV)) < 0.01:
            break
        elif np.linalg.norm(uV) > np.linalg.norm(vV):
            high = mid_y
        else:
            low = mid_y
    return np.array([x, y, mid_x, mid_y])


def draw_region(frame, start_pt, end_pt, regions=3):
    '''
    :param start_pt:
    :param end_pt:
    :param regions: number of desired regions
    :return:
    '''
    # draw regions
    boxLength = abs(start_pt[1][1] - end_pt[1][1])
    num_regions = regions
    eps = boxLength - ((boxLength // num_regions) * num_regions)
    # splits into multiple region
    for step in np.arange(0, boxLength + eps, boxLength // num_regions):
        region_pts = find_equal_line(start_pt, end_pt, start_pt[1][1] - step)
        x_right, y_right, x_left, y_left = region_pts.reshape(4)
        cv2.line(frame, (x_left, y_left), (x_right, y_right), (0, 255, 255), 1)


def check_region(point, start_pt, end_pt, regions=3):
    '''
    :param start_pt:
    :param end_pt:
    :param regions: number of desired regions
    :return:
    '''
    # check regions
    counter = 0
    prev_x_right, prev_y_right, prev_x_left, prev_y_left = start_pt[1][0], start_pt[1][1], start_pt[0][0], start_pt[0][
        1]
    boxLength = abs(start_pt[1][1] - end_pt[1][1])
    num_regions = regions
    eps = boxLength - ((boxLength // num_regions) * num_regions)
    # print(boxLength // num_regions)
    # splits into multiple region
    for step in np.arange(0, boxLength + eps, boxLength // num_regions):
        region_pts = find_equal_line(start_pt, end_pt, start_pt[1][1] - step)
        x_right, y_right, x_left, y_left = region_pts.reshape(4)
        # uV = (x_right - x_left, y_right - y_left)
        # print(np.linalg.norm(uV))
        if step == 0:
            prev_x_right, prev_y_right, prev_x_left, prev_y_left = x_right, y_right, x_left, y_left
        else:
            roi_corners = [
                [prev_x_left, prev_y_left], [x_left, y_left], [x_right, y_right], [prev_x_right, prev_y_right]
            ]
            path = mplPath.Path(roi_corners)
            if path.contains_point(point):
                return counter

            prev_x_right, prev_y_right, prev_x_left, prev_y_left = x_right, y_right, x_left, y_left
            counter += 1


def shortest_distance(right_extreme, lines):
    shortest = 100
    point = None
    index = 0
    for i, line in enumerate(lines):
        tan_point = point_on_tan_line(right_extreme, line)
        # remove points on the left
        if (tan_point[0] - right_extreme[0]) < 0:
            pass
        else:
            uV = (tan_point[0] - right_extreme[0], tan_point[1] - right_extreme[1])
            distance = np.linalg.norm(uV)
            if shortest > distance:
                shortest = distance
                point = tan_point
                index = i

    return index, point, shortest


def main():
    global frame_counter, fps, counter, data, car_count, time_start, start_pt, end_pt, num_region
    while True:
        if frame_counter == 0:
            time_start = time.time()
        frame_counter += 1  # measure fps

        success, frame = cap.read()
        if not success:
            print("***ERROR*** Can't read frame")
            break
        frame = cv2.resize(frame, (W // 2, H // 2))  # resize for better fps
        roi_img = roi(frame, roi_corners)
        fgMask = fgbg.apply(roi_img)
        _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        cnt_img = fgMask.copy()
        cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_GRAY2BGR)
        cnt_img1 = fgMask.copy()
        cnt_img1 = cv2.cvtColor(cnt_img1, cv2.COLOR_GRAY2BGR)

        draw_region(frame, start_pt, end_pt, num_region)
        # find contours
        contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        center = None
        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= 35) and (
                    h >= 35)
            if not contour_valid:
                continue
            else:

                epsilon = 0.000001 * cv2.arcLength(contour, False)
                contour = cv2.approxPolyDP(contour, epsilon, False)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if center is not None:
                    cv2.rectangle(cnt_img1, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cnt_area = cv2.contourArea(contour)
                    cnt_img = cv2.drawContours(cnt_img, contour, -1, (0, 0, 255))
                    # 1. [:,:,0]: take only x value
                    # 2. get the index of the highest x value (aka right extreme)
                    indice = np.argmax(contour[:, :, 0])
                    # right_extreme x coordinate
                    right_extreme_x = contour[indice][0][0]
                    # right_extreme coordinate
                    right_extreme = contour[indice][0]
                    if right_extreme_x > 320:
                        pass
                    else:
                        cv2.circle(frame, (right_extreme[0], right_extreme[1]), 4, (0, 0, 255), -1)
                        lane_index, tan_point, distance = shortest_distance(right_extreme, lines)
                        line_index = check_region(right_extreme, start_pt, end_pt, num_region)
                        if tan_point is not None:
                            cv2.circle(frame, (tan_point[0], tan_point[1]), 4, (0, 255, 255), -1)
                            cv2.putText(frame, format(distance, '.2f'), (tan_point[0], tan_point[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, 0, 1)
                            # cv2.putText(frame, str(line_index), (tan_point[0], tan_point[1] - 10),
                            #             cv2.FONT_HERSHEY_COMPLEX,
                            #             0.4, 0, 1)
                        car_count += 1
                        data.append((cnt_area, lane_index, line_index, distance))

        # get time elapsed until 1 second, compute FPS
        time_elapsed = time.time() - time_start
        if time_elapsed >= 1.0:
            fps = frame_counter
            frame_counter = 0

        cv2.putText(frame, "FPS: " + str(fps), (10, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 2)

        cv2.imshow('cnt_img', cnt_img)
        cv2.imshow('cnt_img1', cnt_img1)
        cv2.imshow('frame', frame)
        cv2.imshow('roi_img', roi_img)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        if k == ord('p'):
            cv2.waitKey(0)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()

    # store in csv file
    count = []
    area_col = []
    lane_index = []
    line_index = []
    dist_col = []

    for datum in data:
        area_col.append(datum[0])
        lane_index.append(datum[1])
        line_index.append(datum[2])
        dist_col.append(datum[3])

    for i in range(car_count):
        count.append(i + 1)
    d = {'Index': count, 'Area': area_col, 'Lane Index': lane_index, 'Line Index': line_index, 'Distance': dist_col}
    df = pd.DataFrame(data=d)
    df.to_csv(path_or_buf='Resources/output_right.csv', index=False)


if __name__ == '__main__':
    main()
