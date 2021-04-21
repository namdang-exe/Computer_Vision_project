import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('car.mp4')
_, frame = cap.read()
H = frame.shape[0]
W = frame.shape[1]
backSub = cv2.createBackgroundSubtractorMOG2(500, 50, True)
frame = cv2.resize(frame, (W // 2, H // 2))

while True:
    success, frame = cap.read()
    if not success:
        break
    # try grabcut
    frame = cv2.resize(frame, (W // 2, H // 2))
    # img = cv2.imread('ratio.png')
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("headlines.jpg", 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(grey, template, cv2.TM_CCOEFF_NORMED)
    # plt.hist(res)
    # plt.show()
    threshold = 0.5
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    cv2.imshow('template', template)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
