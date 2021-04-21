import cv2
import numpy as np

# Read image
im_in = cv2.imread("split.jpg", 0)
# kernel = np.ones((7, 7), np.uint8)
#
# img2 = cv2.morphologyEx(im_in, cv2.MORPH_OPEN, kernel)
canny = cv2.Canny(im_in, 0, 400, apertureSize=3)
cv2.imshow("canny Image", canny)
im2 = cv2.addWeighted(im_in, 0.5, canny, 1, 1)
cv2.imshow('im2', im2)
# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

th, im_th = cv2.threshold(im2, 100, 255, cv2.THRESH_BINARY)


# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

# Display images.
cv2.imshow("Origin Image", im_in)
cv2.imshow("Thresholded Image", im_th)
cv2.imshow("Floodfilled Image", im_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out)
cv2.waitKey(0)
