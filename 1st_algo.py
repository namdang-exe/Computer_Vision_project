import cv2
import numpy as np
import math
import time
import pandas as pd
import matplotlib.path as mplPath
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('car5.mp4')
_, frame = cap.read()
W = frame.shape[1]
H = frame.shape[0]
frame = cv2.resize(frame, (W // 2, H // 2))
first_frame = frame.copy()
height = frame.shape[0]
width = frame.shape[1]
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

roi_corners = [
    [[120, 300], [160, 200], [280, 200], [295, 300]],
    [[450, 300], [350, 200], [475, 200], [625, 300]]
]
data = []
frame_counter = 0
fps = 0
counter = 0
car_count = 0
moments = []
angle_array = []


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


def main():
    global frame_counter, fps, counter, car_count, time_start, data, moments, angle_array
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
        fgMask = filter_mask(fgMask)
        cnt_img = fgMask.copy()
        cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_GRAY2BGR)
        cnt_img1 = fgMask.copy()
        cnt_img1 = cv2.cvtColor(cnt_img1, cv2.COLOR_GRAY2BGR)

        # find contours
        contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        center = None
        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= 25) and (
                    h >= 25) and (w <= 60) and (h <= 60)
            if not contour_valid:
                continue
            else:
                epsilon = 0.000001 * cv2.arcLength(contour, False)
                contour = cv2.approxPolyDP(contour, epsilon, False)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if center is not None:
                    moments.append(center)
                    cv2.circle(first_frame, center, 3, (0, 0, 255), -1)

        # get time elapsed until 1 second, compute FPS
        time_elapsed = time.time() - time_start
        if time_elapsed >= 1.0:
            fps = frame_counter
            frame_counter = 0

        cv2.putText(frame, "FPS: " + str(fps), (10, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 2)
        cv2.imshow('f_frame', first_frame)
        cv2.imshow('cnt_img', cnt_img)
        cv2.imshow('cnt_img1', cnt_img1)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        if k == ord('p'):
            cv2.waitKey(0)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()

    moment_arr = np.array(moments)
    # angle_arr = np.array(angle_array)

    # DBSCAN - clustering algo
    db = DBSCAN(eps=5, min_samples=25).fit(moment_arr)
    # eps: the maximum distance between two samples for one to be considered as in the neighborhood of the other
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    y_db = db.fit_predict(moment_arr)
    # print(y_db)
    # display
    plt.scatter(moment_arr[y_db == 0, 0], moment_arr[y_db == 0, 1], c='red')
    plt.scatter(moment_arr[y_db == 1, 0], moment_arr[y_db == 1, 1], c='black')
    plt.scatter(moment_arr[y_db == 2, 0], moment_arr[y_db == 2, 1], c='blue')
    plt.scatter(moment_arr[y_db == 3, 0], moment_arr[y_db == 3, 1], c='cyan')
    plt.scatter(moment_arr[y_db == 4, 0], moment_arr[y_db == 4, 1], c='olive')
    plt.scatter(moment_arr[y_db == 5, 0], moment_arr[y_db == 5, 1], c='purple')
    # plt.scatter(moment_arr[y_db == 6, 0], moment_arr[y_db == 6, 1], c='brown')
    # plt.scatter(moment_arr[y_db == 7, 0], moment_arr[y_db == 7, 1], c='green')
    # plt.scatter(moment_arr[y_db == 8, 0], moment_arr[y_db == 8, 1], c='orange')
    plt.gca().invert_yaxis()

    # put all angles in an array
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    for i in range(1, len(y_db)):
        if y_db[i] == 0:
            list1.append(angle_array[i])
        elif y_db[i] == 1:
            list2.append(angle_array[i])
        elif y_db[i] == 2:
            list3.append(angle_array[i])
        elif y_db[i] == 3:
            list4.append(angle_array[i])
        elif y_db[i] == 4:
            list5.append(angle_array[i])
        elif y_db[i] == 5:
            list6.append(angle_array[i])

    print('red', np.mean(np.array(list1)))
    print('black', np.mean(np.array(list2)))
    print('blue', np.mean(np.array(list3)))
    print('cyan', np.mean(np.array(list4)))
    print('olive', np.mean(np.array(list5)))
    print('purple', np.mean(np.array(list6)))

    plt.show()


if __name__ == '__main__':
    main()
