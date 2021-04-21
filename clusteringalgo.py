from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def train_bg_subtractor(inst, stream, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print('Training background subtractor')
    i = 0
    while i <= num:
        (grabbed, frame) = stream.read()
        inst.apply(frame, None, 0.001)
        i += 1
    print('Training done!')
    return stream


def filter_mask(img, a=None):
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel2)
    # Remove noise
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
    # Dilate to merge adjacent blobs
    dilation = cv.dilate(opening, kernel2, iterations=2)
    return dilation


parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='car.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
capture = cv.VideoCapture(args.input)

if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

backSub = cv.createBackgroundSubtractorMOG2(history=1000, varThreshold=50, detectShadows=True)
stream = train_bg_subtractor(backSub, capture, num=1000)

# List to keep track of all centroids
pts = []

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    # shadow removal
    _, fgMask = cv.threshold(fgMask, 127, 255, cv.THRESH_BINARY)

    # cv.imshow('Shadow removal', fgMask)
    fgMask = filter_mask(fgMask)

    contours, hierarchy = cv.findContours(
        fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1
    )
    center = None

    # Find car contour and its centroid
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)

        M = cv.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            contour_valid = (w >= 45) and (
                    h >= 45)  # The rectangle box bounds the car has to be big enough

            if not contour_valid:
                continue
            else:
                # cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                # cv.circle(frame, center, 5, (0, 0, 255), -1)
                pts.append(center)

    mask = np.zeros_like(frame)

    # draw centroid
    for i in np.arange(1, len(pts)):

        if pts[i - 1] is None or pts[i] is None:
            continue
        else:
            cv.circle(mask, pts[i], 3, 255, -1)

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('mask', mask)

    keyboard = cv.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break
    elif keyboard == ord('c'):
        cv.imwrite('image1.jpg', frame)

array = np.array(pts)  # turn list of point into array

# DBSCAN
db = DBSCAN(eps=20)
clustering = db.fit(array)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
y_db = db.fit_predict(array)
# print(y_db)
# display
plt.scatter(array[y_db == 0, 0], array[y_db == 0, 1], c='red')
plt.scatter(array[y_db == 1, 0], array[y_db == 1, 1], c='black')
plt.scatter(array[y_db == 2, 0], array[y_db == 2, 1], c='blue')
plt.scatter(array[y_db == 3, 0], array[y_db == 3, 1], c='cyan')
# if len(set(labels)) - (1 if -1 in labels else 0) >= 5:
#     plt.scatter(array[y_db == 4, 0], array[y_db == 4, 1], c='yellow')
#     plt.scatter(array[y_db == 5, 0], array[y_db == 5, 1], c='green')
plt.gca().invert_yaxis()
plt.show()

capture.release()
cv.destroyAllWindows()
