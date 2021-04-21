import math

import cv2
import numpy as np
from statistics import mean


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def find_theta(pt1, pt2):
    # find theta angle when input two points
    # assume car is approaching the camera
    # p1 == starting point | pt2 == ending point
    # theta = None
    # a = None
    x1, y1 = pt1
    x2, y2 = pt2
    dX = x2 - x1
    dY = y2 - y1
    if math.isnan(dX) or math.isnan(dY):
        return None
    elif dY < 5:  # wrong direction lines (aka noise)
        return None
    elif dY < 20:  # horizontal lines
        return 180
    elif abs(dX) < 5:  # vertical line
        return 90
    elif dX < 0:
        return np.degrees(np.arctan(abs(dY / dX)))
    elif dX > 0:
        return 180 - np.degrees(np.arctan(abs(dY / dX)))
    else:
        return None

    # # calculate theta using dot product ab = |a||b|cos(θ)
    # uV = np.array([(x1 - x2), (y1 - y2)])
    # vV = np.array([(x1 - x2), (y2 - y2)])
    # dot = np.dot(uV, vV)
    # u_len = np.linalg.norm(uV)
    # v_len = np.linalg.norm(vV)
    # # finds the angle
    # if u_len * v_len != 0:
    #     a = math.acos(dot / (u_len * v_len))
    # # theta in radian
    # if a is not None and not math.isnan(a):
    #     theta = int(a * (180 / math.pi))
    # if y1 > y2:
    #     return None
    # elif abs(y1 - y2) < 20:
    #     return 180
    # elif abs(x1 - x2) < 10:
    #     return 90
    # elif theta is not None:
    #     if x1 > x2:
    #         return theta
    #     else:
    #         return 180 - theta
    # else:
    #     return None


def display_flow(img, flow, stride=40, win_name='optical flow'):
    # mask = np.zeros_like(img)
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i * stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10 * delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1], (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)
            theta_angle = find_theta(pt1[::-1], pt2[::-1])
            # if theta_angle is not None and theta_angle != 0:
            #     print("theta of " + win_name + ": ", theta_angle)
            # cv2.circle(mask, pt2[::-1], 3, (255, 0, 0), -1)

    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow(win_name, img)
    # cv2.imshow(win_name + ' mask', mask)
    # cv2.imshow('norm_opt_flow', norm_opt_flow)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        return 1
    else:
        return 0


def normalizeAngle(angle):
    newAngle = angle % 360
    newAngle = (newAngle + 360) % 360
    if newAngle > 180:
        newAngle -= 360
    return newAngle


def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(0, 626), (220, height // 2), (480, height // 2), (400, height)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


cap = cv2.VideoCapture('car.mp4')
_, prev_frame = cap.read()
# resize for better speed
# prev_frame = cv2.resize(prev_frame, (0, 0), None, 0.5, 0.5)
height = prev_frame.shape[0]
width = prev_frame.shape[1]
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# prev_frame1 = prev_frame[0:height // 3, 0: width]
# prev_frame2 = prev_frame[height // 3:2 * height // 3, 0: width]
# prev_frame3 = prev_frame[2 * height // 3:height, 0: width]
prev_frame3 = roi(prev_frame)
init_flow = True
angles = []

while True:
    status_cap, frame = cap.read()
    if not status_cap:
        break
    # frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray1 = gray[0:height // 3, 0:width]
    # gray2 = gray[height // 3:2 * height // 3, 0:width]
    gray4 = gray[2 * height // 3:height, 0: width]
    gray3 = roi(gray)
    if init_flow:
        # opt_flow1 = cv2.calcOpticalFlowFarneback(prev_frame1, gray1, None, 0.5, 5, 13, 10, 5, 1.1,
        #                                          cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        #
        # opt_flow2 = cv2.calcOpticalFlowFarneback(prev_frame2, gray2, None, 0.5, 5, 13, 10, 5, 1.1,
        #                                          cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        opt_flow3 = cv2.calcOpticalFlowFarneback(prev_frame3, gray3, None, 0.5, 3, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        init_flow = False
    else:
        # opt_flow1 = cv2.calcOpticalFlowFarneback(prev_frame1, gray1, opt_flow1, 0.5, 5, 13, 10, 5, 1.1,
        #                                          cv2.OPTFLOW_USE_INITIAL_FLOW)
        # opt_flow2 = cv2.calcOpticalFlowFarneback(prev_frame2, gray2, opt_flow2, 0.5, 5, 13, 10, 5, 1.1,
        #                                          cv2.OPTFLOW_USE_INITIAL_FLOW)
        opt_flow3 = cv2.calcOpticalFlowFarneback(prev_frame3, gray3, opt_flow3, 0.5, 3, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_USE_INITIAL_FLOW)

    # magnitude1, angle1 = cv2.cartToPolar(opt_flow1[..., 0], opt_flow1[..., 1], angleInDegrees=True)
    # # print(angle1)
    # angle1 = np.int0(np.reshape(angle1, angle1.size))
    # # print("theta of 1: ", np.bincount(angle1).argmax())
    # print("theta of 1: ", normalizeAngle(np.bincount(angle1).argmax()))
    # magnitude2, angle2 = cv2.cartToPolar(opt_flow2[..., 0], opt_flow2[..., 1], angleInDegrees=True)
    # angle2 = np.int0(np.reshape(angle2, angle2.size))
    # # print("theta of 2: ", np.bincount(angle2).argmax())
    # print("theta of 2: ", normalizeAngle(np.bincount(angle2).argmax()))
    # magnitude3, angle3 = cv2.cartToPolar(opt_flow3[..., 0], opt_flow3[..., 1], angleInDegrees=True)
    # angle3 = np.int0(np.reshape(angle3, angle3.size))
    # # print("theta of 3: ", np.bincount(angle3).argmax())
    # print("theta of 3: ", normalizeAngle(np.bincount(angle3).argmax()))

    # prev_frame1 = np.copy(gray1)
    # prev_frame2 = np.copy(gray2)
    prev_frame3 = np.copy(gray3)

    # for index in np.ndindex(opt_flow3[::40, ::40].shape[:2]):
    #     pt1 = tuple(i * 40 for i in index)
    #     delta = opt_flow3[pt1].astype(np.int32)[::-1]
    #     pt2 = tuple(pt1 + 10 * delta)
    #     if 2 <= cv2.norm(delta) <= 10:
    #         theta_angle = find_theta(pt1[::-1], pt2[::-1])
    #         if theta_angle is not None and 100 > theta_angle > 40:
    #             theta_angle = int(theta_angle)
    #             cv2.arrowedLine(gray3, pt1[::-1], pt2[::-1], (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)
    #             cv2.putText(gray3, str(theta_angle), (pt1[::-1][0] + 10, pt1[::-1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.6,
    #                         (0, 0, 255), 3)
    #             print(theta_angle)
    #             angles.append(theta_angle)
    draw_flow(gray3, opt_flow3)
    cv2.imshow('flow', gray3)
    # cv2.imshow('3', gray3)
    # cv2.imshow('4', gray4)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
    # if display_flow(gray1, opt_flow1, win_name='1') or display_flow(gray2, opt_flow2, win_name='2') or display_flow(
    #         gray3, opt_flow3, win_name='3'):
    #     break

# array = np.array(angles)
# print('The mean theta is: ', mean(array))

cap.release()
cv2.destroyAllWindows()
