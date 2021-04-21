import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass


cv2.namedWindow('Canny')
cv2.createTrackbar('A', 'Canny', 0, 250, nothing)
cv2.createTrackbar('B', 'Canny', 150, 400, nothing)

while True:
    img = cv2.imread('background.png')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = np.zeros_like(blur)
    # polygon = np.array([
    #     [(180, 250), (200, 100), (305, 100), (320, 250)]
    # ])
    # cv2.fillPoly(mask, polygon, (255,255,255))
    # masked_image = cv2.bitwise_or(blur, mask)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    lower = cv2.getTrackbarPos('A', 'Canny')
    upper = cv2.getTrackbarPos('B', 'Canny')

    canny = cv2.Canny(blur, lower, upper, apertureSize=3)
    cv2.imshow('Canny', canny)
    # plt.imshow(masked_image)
    # plt.show()

cv2.destroyAllWindows()
