import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    # print(fx)
    # print(np.int32(fx))
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int0(lines + 0.5)
    # print(lines)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 0, 255))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


cap = cv2.VideoCapture('car4.mp4')
_, prev_frame = cap.read()

height = prev_frame.shape[0]
width = prev_frame.shape[1]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.resize(prev_gray, (width // 2, height // 2))
data = []
while True:
    status_cap, frame = cap.read()
    if not status_cap:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width // 2, height // 2))
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang = angle * 180 / np.pi  # conversion to degree
    # print(np.int0(ang))
    data.append(np.int0(ang))

    # print(ang[index])
    prev_gray = gray.copy()

    flow_image = draw_flow(gray, flow, step=16)
    cv2.imshow('flow', flow_image)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
    if k == ord('p'):
        cv2.waitKey(0)

# plt.imshow(flow_image)
# plt.gca().invert_xaxis()
# plt.show()
cv2.imshow('frame', flow_image)
cv2.waitKey(0)
data = data[len(data) - 1]
pd.DataFrame(data).to_csv("angle.csv", index=False)
cap.release()
cv2.destroyAllWindows()
