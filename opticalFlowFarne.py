import math

import cv2
import numpy as np


def find_theta(pt1, pt2):
    # find theta angle when input two points
    # assume car is approaching the camera
    # p1 == starting point | pt2 == ending point
    theta = None
    a = None
    x1, y1 = pt1
    x2, y2 = pt2
    if abs(x1-x2) < 0.1 and y1 != y2:
        return 90
    # calculate theta using dot product ab = |a||b|cos(θ)
    uV = np.array([(x1 - x2), (y1 - y2)])
    vV = np.array([(x1 - x2), (y2 - y2)])
    dot = np.dot(uV, vV)
    u_len = np.linalg.norm(uV)
    v_len = np.linalg.norm(vV)
    # angle in radian
    if u_len * v_len != 0:
        a = math.acos(dot / (u_len * v_len))
    # theta
    if a is not None and not math.isnan(a):
        theta = int(a * (180 / math.pi))
    if theta is not None:
        return theta


def normalizeAngle(angle):
    # Normalize angle in range of -179 and 180 degree
    newAngle = angle % 360
    newAngle = (newAngle + 360) % 360
    if newAngle > 180:
        newAngle -= 360
    return newAngle


cap = cv2.VideoCapture('car.mp4')
_, prev_frame = cap.read()
# resize for better speed
prev_frame = cv2.resize(prev_frame, (0, 0), None, 0.5, 0.5)
height = prev_frame.shape[0]
width = prev_frame.shape[1]
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# split frame into 3
prev_frame1 = prev_frame[0:height // 3, 0: width]
prev_frame2 = prev_frame[height // 3:2 * height // 3, 0: width]
prev_frame3 = prev_frame[2 * height // 3:height, 0: width]
init_flow = True

while True:
    status_cap, frame = cap.read()
    if not status_cap:
        break
    frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # splits frame into 3
    gray1 = gray[0:height // 3, 0:width]
    gray2 = gray[height // 3:2 * height // 3, 0:width]
    gray3 = gray[2 * height // 3:height, 0: width]
    # calculates the first optical flow from the first frame
    if init_flow:
        opt_flow1 = cv2.calcOpticalFlowFarneback(prev_frame1, gray1, None, 0.5, 5, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        opt_flow2 = cv2.calcOpticalFlowFarneback(prev_frame2, gray2, None, 0.5, 5, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        opt_flow3 = cv2.calcOpticalFlowFarneback(prev_frame3, gray3, None, 0.5, 5, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        init_flow = False
    else:
        opt_flow1 = cv2.calcOpticalFlowFarneback(prev_frame1, gray1, opt_flow1, 0.5, 5, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_USE_INITIAL_FLOW)
        opt_flow2 = cv2.calcOpticalFlowFarneback(prev_frame2, gray2, opt_flow2, 0.5, 5, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_USE_INITIAL_FLOW)
        opt_flow3 = cv2.calcOpticalFlowFarneback(prev_frame3, gray3, opt_flow3, 0.5, 5, 13, 10, 5, 1.1,
                                                 cv2.OPTFLOW_USE_INITIAL_FLOW)

    # From the algorithm we can extract the magnitude and angle of the motion
    # magnitude1, angle1 = cv2.cartToPolar(opt_flow1[..., 0], opt_flow1[..., 1], angleInDegrees=True)
    # # angle1 returns an array of all angles where motion has been detected
    #
    # # convert angle to integer
    # angle1 = np.int0(np.reshape(angle1, angle1.size))
    # print("theta of 1: ", normalizeAngle(np.bincount(angle1).argmax()))
    # # np.bincount(angle1).argmax(): most frequent angle
    # magnitude2, angle2 = cv2.cartToPolar(opt_flow2[..., 0], opt_flow2[..., 1], angleInDegrees=True)
    # angle2 = np.int0(np.reshape(angle2, angle2.size))
    # print("theta of 2: ", normalizeAngle(np.bincount(angle2).argmax()))
    # magnitude3, angle3 = cv2.cartToPolar(opt_flow3[..., 0], opt_flow3[..., 1], angleInDegrees=True)
    # angle3 = np.int0(np.reshape(angle3, angle3.size))
    # print("theta of 3: ", normalizeAngle(np.bincount(angle3).argmax()))

    prev_frame1 = np.copy(gray1)
    prev_frame2 = np.copy(gray2)
    prev_frame3 = np.copy(gray3)

    # displays arrows
    for index in np.ndindex(opt_flow3[::40, ::40].shape[:2]):
        pt1 = tuple(i * 40 for i in index)
        delta = opt_flow3[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10 * delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(gray3, pt1[::-1], pt2[::-1], (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)
            theta_angle = find_theta(pt1[::-1], pt2[::-1])
            if theta_angle is not None:
                cv2.putText(gray3, str(theta_angle), (pt1[::-1][0] + 10, pt1[::-1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (0, 0, 255), 3)

    cv2.imshow('3', gray3)
    k = cv2.waitKey(50) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
