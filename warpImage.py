import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('frame.jpg')
height = img.shape[0]

# (0, 626), (220, height // 2), (480, height // 2), (400, height)
# source points
top_left = [220, height // 2]
top_right = [490, height // 2]
bottom_right = [400, height]
bottom_left = [0, 626]
pts = np.array([bottom_left, bottom_right, top_right, top_left])

# target points
# y_off = 400  # y offset
top_left_dst = [top_left[0], top_left[1]]
top_right_dst = [top_left_dst[0] + 270, top_left_dst[1]]  # top_right = top_left + lane w
bottom_right_dst = [top_right_dst[0], top_right_dst[1] + height // 2]
bottom_left_dst = [top_left_dst[0], bottom_right_dst[1]]
dst_pts = np.array([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])

# generate a preview to show where the warped bar would end up
preview = np.copy(img)
cv2.polylines(preview, np.int32([dst_pts]), True, (0, 0, 255), 5)
cv2.polylines(preview, np.int32([pts]), True, (255, 0, 255), 1)


# calculate transformation matrix
pts = np.float32(pts.tolist())
dst_pts = np.float32(dst_pts.tolist())
M = cv2.getPerspectiveTransform(pts, dst_pts)

# wrap image and draw the resulting image
image_size = (img.shape[1], img.shape[0])
cv2.rectangle(img, (300, 239), (360, 272), (255, 0, 0), 3)
warped = cv2.warpPerspective(img, M, (bottom_right_dst[1], top_right_dst[0]))

cv2.imshow("preview", preview)
cv2.imshow('img', img)
cv2.imshow('warped', warped)
# back = np.ones(img)


cv2.waitKey(0)
cv2.destroyAllWindows()
