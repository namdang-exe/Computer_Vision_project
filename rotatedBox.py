import numpy as np
import cv2
import matplotlib.pyplot as plt


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def unwarp(img, roi_corners):
    '''
    Unwarp image using 4 points and getPerspectiveTransform
    '''
    src = np.float32(roi_corners)
    #
    warped_size = (roi_corners[2][0] - roi_corners[1][0], roi_corners[0][1] - roi_corners[1][1])
    # dst
    dst = np.float32([[0, warped_size[1]],
                      [0, 0],
                      [warped_size[0], 0],
                      [warped_size[0], warped_size[1]]])

    Mpersp = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Mpersp, dsize=warped_size)
    return warped


# roi_corners = [[0, 626],  # left, down
#                [220, height // 2],  # left, up
#                [480, height // 2],  # right, up
#                [400, height]]  # right, down
img = cv2.imread('frame.jpg')
drawImg = img.copy()
height = img.shape[0]
width = img.shape[1]

rotated_box = np.array([
    [[0, 606], [150, 434], [480, 500], [420, height]]
])
roi_corners = [[0, 606], [150, 434], [480, 500], [420, height]]
cv2.polylines(img, rotated_box, True, (0, 255, 0), 3)
# rot_rect = ((x,y), (width, height), angle)
res = unwarp(img, roi_corners)
cv2.imshow('res', res)
cv2.rectangle(drawImg, (0, 493), (442, height), (0, 0, 255), 3)
cv2.imshow('drawImg', drawImg)
cv2.rectangle(img, (0, 493), (442, height), (0, 0, 255), 3)
block1 = drawImg[493:height, 0:442]
cv2.imshow('block1', block1)
imgStack = stackImages(1, ([block1], [block1], [res]))
cv2.imshow('img', img)
# cv2.imshow('imgStack', imgStack)


cv2.waitKey(0)
cv2.destroyAllWindows()
