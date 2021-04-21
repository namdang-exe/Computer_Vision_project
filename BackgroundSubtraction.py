from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

from scipy.spatial import distance
from collections import deque
import imutils


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


def filter_mask(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv.dilate(opening, kernel, iterations=2)
    return dilation


# def watershed_algo(image):
#     D = ndimage.distance_transform_edt(image)
#     localMax = peak_local_max(D, indices=False, min_distance=50,
#                               labels=image)
#     # perform a connected component analysis on the local peaks,
#     # using 8-connectivity, then apply the Watershed algorithm
#     markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
#     labels = watershed(-D, markers, mask=image)
#     # loop over the unique labels returned by the Watershed
#     # algorithm
#     for label in np.unique(labels):
#         # if the label is zero, we are examining the 'background'
#         # so simply ignore it
#         if label == 0:
#             continue
#         # otherwise, allocate memory for the label region and draw
#         # it on the mask
#         mask = np.zeros(image.shape, dtype="uint8")
#         mask[labels == label] = 255
#         # detect contours in the mask and grab the largest one
#         cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
#                                cv.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         c = max(cnts, key=cv.contourArea)
#         # draw a circle enclosing the object
#         ((x, y), r) = cv.minEnclosingCircle(c)
#         cv.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
#         cv.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def take_second(elem):
    return elem[1]


def displays_line(image, pts=None, n_clusters='auto', line_length=500, road_width=140):
    pass


parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='car.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(history=1000, varThreshold=50, detectShadows=True)
else:
    backSub = cv.createBackgroundSubtractorKNN(history=1000, dist2Threshold=1, detectShadows=True)

capture = cv.VideoCapture(args.input)

# take first frame
_, frame1 = capture.read()
height = frame1.shape[0]
width = frame1.shape[1]

if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

stream = train_bg_subtractor(backSub, capture, num=1000)

# p0 = p0.reshape(-1, 1, 2)

pts = []
# (dX, dY) = (0, 0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)
    fgMask = backSub.apply(frame)
    # cv.imshow('BF Shadow removal', fgMask)
    # shadow removal
    _, fgMask = cv.threshold(fgMask, 127, 255, cv.THRESH_BINARY)

    # cv.imshow('Shadow removal', fgMask)
    fgMask = filter_mask(fgMask)
    # fgMask = cv.GaussianBlur(fgMask, (5, 5), 0)

    # calculate optical flow

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), (255, 0, 0), 3)

    contours, hierarchy = cv.findContours(
        fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1
    )
    center = None

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)

        M = cv.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            contour_valid = (w >= 55) and (
                    h >= 55)

            if not contour_valid:
                continue
            else:
                # draw rotated rects
                # rect = cv.minAreaRect(contour)
                # box = cv.boxPoints(rect)
                # box = np.int0(box)
                # cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
                # cv.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 3)
                # cv.circle(frame, center, 5, (0, 0, 255), -1)
                pts.append(center)

    # polygons = np.array([
    #     [(17, 589), (251, 313), (500, 376), (420, 700)]
    # ])
    # cv.polylines(frame, polygons, True, (0, 255, 0), 3)

    # draw centroid
    # for i in range(len(pts)):
    #
    #     if pts[i - 1] is None or pts[i] is None:
    #         continue
    #     else:
    #         cv.circle(mask, pts[i], 3, 255, -1)

    # draw lines
    # for i in np.arange(1, len(pts)):
    #     dX = pts[i - 1][0] - pts[i][0]
    #     dY = pts[i - 1][1] - pts[i][1]
    #     if pts[i - 1] is None or pts[i] is None:
    #         continue
    #     # elif abs(dY - dX) < 0.73:
    #     else:
    #         cv.line(mask, pts[i - 1], pts[i], 255, 3)

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('mask', mask)

    keyboard = cv.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break
    elif keyboard == ord('c'):
        cv.imwrite('image1.jpg', frame)

    # Now update the previous frame and previous points
    old_gray = fgMask.copy()
    p0 = good_new.reshape(-1, 1, 2)

array = np.array(pts)
# # kmeans
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(array)
# y_km = kmeans.fit_predict(array)
#
# plt.scatter(array[y_km == 0, 0], array[y_km == 0, 1], s=100, c='red')
# plt.scatter(array[y_km == 1, 0], array[y_km == 1, 1], s=100, c='black')
# plt.scatter(array[y_km == 2, 0], array[y_km == 2, 1], s=100, c='blue')
# plt.scatter(array[y_km == 3, 0], array[y_km == 3, 1], s=100, c='cyan')
# plt.gca().invert_yaxis()
# plt.show()

# # hierarchy clustering
# hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')
# y_hc = hc.fit_predict(array)
# plt.scatter(array[y_hc == 0, 0], array[y_hc == 0, 1], s=100, c='red')
# plt.scatter(array[y_hc == 1, 0], array[y_hc == 1, 1], s=100, c='black')
# plt.scatter(array[y_hc == 2, 0], array[y_hc == 2, 1], s=100, c='blue')
# plt.scatter(array[y_hc == 3, 0], array[y_hc == 3, 1], s=100, c='cyan')
# plt.gca().invert_yaxis()
# plt.show()

# DBSCAN
db = DBSCAN(eps=27, min_samples=22).fit(array)
# eps: the maximum distance between two samples for one to be considered as in the neighborhood of the other
labels = db.labels_
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


# linear regression
linear_regressor = LinearRegression()  # create object for the class

line_image = frame1.copy()
line_image = cv.cvtColor(line_image, cv.COLOR_BGR2RGB)
# plt.imshow(line_image)
# separate clusters
list1 = []
list2 = []
list3 = []
list4 = []
for i in range(1, len(y_db)):
    if y_db[i] == 0:
        list1.append((array[i][0], array[i][1]))
    elif y_db[i] == 1:
        list2.append((array[i][0], array[i][1]))
    elif y_db[i] == 2:
        list3.append((array[i][0], array[i][1]))
    elif y_db[i] == 3:
        list4.append((array[i][0], array[i][1]))

# sorted_list = sorted(list1, key=take_second)
# sorted_list = np.array(sorted_list)
# n = 4  # number of chunks
# X = sorted_list[:, 0]
# # break into chunks
# x_chunks = list(chunks(X, len(X) // n))
# x_chunks = np.array(x_chunks)
# Y = sorted_list[:, 1]
# y_chunks = list(chunks(Y, len(Y) // n))
# y_chunks = np.array(y_chunks)
# for i in range(len(y_chunks)):
#     # get the coefficients
#     coefs = np.polyfit(x_chunks[i], y_chunks[i], 1)
#     # create the poly function
#     poly = np.poly1d(coefs)
#     new_x = np.linspace(min(x_chunks[i]), max(x_chunks[i]))
#     new_x = np.delete(new_x, np.where(poly(new_x) > height))
#     new_x = np.delete(new_x, np.where(poly(new_x) < 300))
#     new_y = poly(new_x)
#     plt.plot(new_x, new_y, c='black')
#
# plt.gca().invert_yaxis()
# plt.show()
# draw lines on image
# list1 = np.array(list1)
# x = list1[:, 0]
# y = list1[:, 1]
# # get coefficients
# coefs = np.polyfit(x, y, 2)  # y = mx + b
# # create polynomial equation
# poly = np.poly1d(coefs)
# new_x = np.linspace(min(x), max(x))
# # create a limit for the line
# new_x = np.delete(new_x, np.where(poly(new_x) > height))
# new_x = np.delete(new_x, np.where(poly(new_x) < 300))
# new_x = np.int0(new_x)
# new_y = np.int0(poly(new_x))
# mat = list(zip(new_x, new_y))
# mat = np.array(mat)
#
# cv.polylines(line_image, [mat], False, (0, 0, 255), 3)
# # plt.plot(new_x, new_y, color='red')
# # cv.line(line_image, (int(min(new_x)), int(poly(min(new_x)))), (int(max(new_x)), int(poly(max(new_x)))), (255, 0, 0), 4)
#
# # roi
# # roi_x0 = new_x + 70
# # cv.line(line_image, (int(min(roi_x0)), int(poly(min(new_x)))), (int(max(roi_x0)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
# # roi_x1 = new_x - 70
# # cv.line(line_image, (int(min(roi_x1)), int(poly(min(new_x)))), (int(max(roi_x1)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
# # cv.line(line_image, (int(min(roi_x1)), int(poly(min(new_x)))), (int(max(roi_x1)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
# # cv.line(line_image, (int(min(roi_x1)), int(poly(min(new_x)))), (int(max(roi_x1)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
#
# list2 = np.array(list2)
# x = list2[:, 0]
# y = list2[:, 1]
# # get coefficients
# coefs = np.polyfit(x, y, 2)
# # create polynomial equation
# poly = np.poly1d(coefs)
# new_x = np.linspace(min(x), max(x))
# # create a limit for the line
# new_x = np.delete(new_x, np.where(poly(new_x) > height))
# new_x = np.delete(new_x, np.where(poly(new_x) < 300))
# new_x = np.int0(new_x)
# new_y = np.int0(poly(new_x))
# mat = list(zip(new_x, new_y))
# mat = np.array(mat)
# cv.polylines(line_image, [mat], False, (0, 0, 255), 3)
# # plt.plot(new_x, new_y, color='black')
# # cv.line(line_image, (int(min(new_x)), int(poly(min(new_x)))), (int(max(new_x)), int(poly(max(new_x)))), (255, 0, 0), 4)
# # roi
# # roi_x0 = new_x + 70
# # cv.line(line_image, (int(min(roi_x0)), int(poly(min(new_x)))), (int(max(roi_x0)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
# # roi_x1 = new_x - 70
# # cv.line(line_image, (int(min(roi_x1)), int(poly(min(new_x)))), (int(max(roi_x1)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
#
# list3 = np.array(list3)
# x = list3[:, 0]
# y = list3[:, 1]
# # get coefficients
# coefs = np.polyfit(x, y, 2)
# # create polynomial equation
# poly = np.poly1d(coefs)
# new_x = np.linspace(min(x), max(x))
# # create a limit for the line
# new_x = np.delete(new_x, np.where(poly(new_x) > height))
# new_x = np.delete(new_x, np.where(poly(new_x) < 300))
# new_x = np.int0(new_x)
# new_y = np.int0(poly(new_x))
# mat = list(zip(new_x, new_y))
# mat = np.array(mat)
# cv.polylines(line_image, [mat], False, (0, 0, 255), 3)
# # plt.plot(new_x, new_y, color='blue')
# # cv.line(line_image, (int(min(new_x)), int(poly(min(new_x)))), (int(max(new_x)), int(poly(max(new_x)))), (255, 0, 0), 4)
# # roi
# # roi_x0 = new_x + 70
# # cv.line(line_image, (int(min(roi_x0)), int(poly(min(new_x)))), (int(max(roi_x0)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
# # roi_x1 = new_x - 70
# # cv.line(line_image, (int(min(roi_x1)), int(poly(min(new_x)))), (int(max(roi_x1)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
#
# list4 = np.array(list4)
# x = list4[:, 0]
# y = list4[:, 1]
# # get coefficients
# coefs = np.polyfit(x, y, 2)
# # create polynomial equation
# poly = np.poly1d(coefs)
# new_x = np.linspace(min(x), max(x))
# # create a limit for the line
# new_x = np.delete(new_x, np.where(poly(new_x) > height))
# new_x = np.delete(new_x, np.where(poly(new_x) < 300))
# new_x = np.int0(new_x)
# new_y = np.int0(poly(new_x))
# mat = list(zip(new_x, new_y))
# mat = np.array(mat)
# cv.polylines(line_image, [mat], False, (0, 0, 255), 3)
# # plt.plot(new_x, new_y, color='cyan')
# # cv.line(line_image, (int(min(new_x)), int(poly(min(new_x)))), (int(max(new_x)), int(poly(max(new_x)))), (255, 0, 0), 4)
# # roi
# # roi_x0 = new_x + 70
# # cv.line(line_image, (int(min(roi_x0)), int(poly(min(new_x)))), (int(max(roi_x0)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
# # roi_x1 = new_x - 70
# # cv.line(line_image, (int(min(roi_x1)), int(poly(min(new_x)))), (int(max(roi_x1)), int(poly(max(new_x)))), (0, 255, 0),
# #         4)
#
# plt.imshow(line_image)
# # plt.gca().invert_yaxis()
# plt.show()

# matplotlib
# list1 = np.array(list1)
# X = list1[:, 0].reshape(-1, 1)
# Y = list1[:, 1].reshape(-1, 1)
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions
# plt.plot(X, Y_pred, color='red')
#
# list2 = np.array(list2)
# X = list2[:, 0].reshape(-1, 1)
# Y = list2[:, 1].reshape(-1, 1)
#
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions
# plt.plot(X, Y_pred, color='red')
#
# list3 = np.array(list3)
# X = list3[:, 0].reshape(-1, 1)
# Y = list3[:, 1].reshape(-1, 1)
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions
# plt.plot(X, Y_pred, color='red')
#
# list4 = np.array(list4)
# X = list4[:, 0].reshape(-1, 1)
# Y = list4[:, 1].reshape(-1, 1)
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions
# plt.plot(X, Y_pred, color='red')
#
# plt.ylim(0, height)
# plt.gca().invert_yaxis()
# plt.show()

capture.release()
cv.destroyAllWindows()
