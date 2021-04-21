import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import csv


def roi(image, roi_corners):
    mask = np.zeros_like(image)
    for roi_corner in roi_corners:
        polygons = np.array(roi_corner, np.int32)
        polygons = polygons.reshape((-1, 1, 2))
        cv2.polylines(mask, [polygons], True, (255, 255, 255), 3)
        cv2.fillPoly(mask, [polygons], (255, 255, 255))
        masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# construct parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-r", "--roi-on", type=str2bool, default=False, help="True/ False: True = turn on roi")
args = vars(ap.parse_args())

# variables

cap = cv2.VideoCapture(args["input"])
_, frame = cap.read()
W = frame.shape[1]
H = frame.shape[0]
frame = cv2.resize(frame, (W // 2, H // 2))
width = frame.shape[1]
height = frame.shape[0]

roi_corners = [
    [[0, height - 3], [129, 115], [191, 115], [161, height]],
    [[256, height], [226, 115], [286, 115], [407, height]]
]

roi_img = roi(frame, roi_corners)
if args["roi_on"]:
    plt.imshow(roi_img)
else:
    plt.imshow(frame)
plt.show()
