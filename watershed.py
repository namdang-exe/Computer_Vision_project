# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2


def watershed_algo(image):
    height = image.shape[0]
    width = image.shape[1]
    mask2 = image.copy()
    color = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    D = ndimage.distance_transform_edt(image)
    localMax = peak_local_max(D, indices=False, min_distance=40,
                              labels=image)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=image)

    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        (x, y, w, h) = cv2.boundingRect(c)

        # split off the frame, max height = car length
        # width max = width/ 2
        contour_valid = (35 <= w <= width) and (
                35 <= h <= height)

        if not contour_valid:
            continue
        else:
            cv2.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return color


image = cv2.imread('split.jpg')
water_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((7, 7), np.uint8)
kernel1 = np.ones((17, 17), np.uint8)
img2 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel1)
# thresh = cv2.threshold(gray, 0, 255,
#                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("img2", img2)

water_image = watershed_algo(img2)

# show the output image
cv2.imshow("water_image", water_image)

cv2.waitKey(0)
