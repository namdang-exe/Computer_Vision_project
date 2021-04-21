import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils


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


cap = cv2.VideoCapture('car.mp4')
_, first_frame = cap.read()
H, W, D = first_frame.shape
mog = cv2.createBackgroundSubtractorMOG2(500, detectShadows=True)
# train_bg_subtractor(mog, cap, 500)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (W // 2, H // 2))
    mog.apply(frame)
    background = mog.getBackgroundImage()
    imgThres, imgCanny, imgColor = utils.thresholding(background)


    cv2.imshow('imgThres', imgThres)
    cv2.imshow('imgCanny', imgCanny)
    cv2.imshow('imgColor', imgColor)
    cv2.imshow('background', background)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
