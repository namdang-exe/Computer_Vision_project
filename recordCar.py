import numpy as np
import cv2
import math
import matplotlib.path as mplPath


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def filter_mask(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)
    return dilation


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def draw_lines(img, angle, rect=None, lines=30):
    '''
    This function is to create a matrix on line on the image to capture car movement
    Matrix lines are red
    If the blob touches the line, the line will turn green
    '''
    if rect is None:
        rect = ((0, 0), (0, 0), 0)
    angle = abs(angle)
    height = img.shape[0]
    width = img.shape[1]
    mask = img.copy()
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    (x, y), (w, h), theta = rect
    y0 = 0
    y1 = height
    for x0 in np.arange(0, width + 60, width // lines):
        dX = int(height / np.tan(np.radians(angle)))
        x1 = x0 - dX
        cv2.line(mask, (x0, y0), (x1, y1), (0, 0, 255), 2)

    for x0 in np.arange(0, width + 60, width // lines):
        dX = int(height / np.tan(np.radians(angle)))
        x1 = x0 - dX
        eff = np.polyfit((x0, x1), (y0, y1), 1)
        poly = np.poly1d(eff)
        if abs(y - poly(x)) <= 5:
            # cv2.line(mask, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for i in np.arange(0, w, width // lines):
                cv2.line(mask, (x0 + int(i) - 15, y0), (x1 + int(i) - 15, y1), (0, 255, 0), 2)
    return mask


def find_equal_line(roi_corners, y):
    """
    This is binary search algorithm to find a line that cross left and right lane
    Both sides on left and right lane are equal
    :param roi_corners:  enter Region of Interest
    :param y: y_value of the point on the left lane
    :return: coords of the equal line
    """
    left_line_parameter = np.polyfit((roi_corners[0][0], roi_corners[1][0]), (roi_corners[0][1], roi_corners[1][1]), 1)
    right_line_parameter = np.polyfit((roi_corners[2][0], roi_corners[3][0]), (roi_corners[2][1], roi_corners[3][1]), 1)
    m1, b1 = left_line_parameter
    m2, b2 = right_line_parameter
    # finds the intercept of two lanes
    x0 = int((b2 - b1) / (m1 - m2))
    y0 = int(m1 * x0 + b1)
    # point on the left lane
    x = int((y - b1) / m1)
    # performs bin search
    uV = np.array([(x0 - x), (y0 - y)])
    high = y0
    low = max(roi_corners[0][1], roi_corners[1][1], roi_corners[2][1], roi_corners[3][1])
    while abs(high - low) > 1:
        mid_y = (high + low) // 2
        mid_x = int((mid_y - b2) / m2)
        vV = np.array([(x0 - mid_x), (y0 - mid_y)])
        if abs(np.linalg.norm(uV) - np.linalg.norm(vV)) < 0.01:
            break
        elif np.linalg.norm(uV) > np.linalg.norm(vV):
            high = mid_y
        else:
            low = mid_y

    return np.array([x, y, mid_x, mid_y])


def draw_roi(img, roi_corners, region=False, lines=None):
    """
    This function is just to DRAW the regions and lines we want to track
    :param img:
    :param roi_corners: Region of Interest
    :param region: If we want to turn on the 3 lines to track car position, region == True
    :param lines: amount of lines we want
    :return: an img with the drawings
    """
    mask = img.copy()
    # polygon
    polygons = np.array(roi_corners, np.int32)
    polygons = polygons.reshape((-1, 1, 2))
    cv2.polylines(mask, [polygons], True, (0, 255, 0), 3)
    if region:
        # draw regions
        boxLength = abs(roi_corners[1][1] - roi_corners[0][1])
        num_regions = lines + 1
        # splits into multiple lines
        for step in np.arange(boxLength // num_regions + 1, boxLength, boxLength // num_regions):
            region_pts = find_equal_line(roi_corners, roi_corners[1][1] + step - 20)
            x0, y0, x1, y1 = region_pts.reshape(4)
            cv2.line(mask, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return mask


def draw_roi_track(img, roi_corners, contour, region=False, lines=None):
    """
    Car Tracking Function
    :param contour:
    :param img:
    :param roi_corners:
    :param box:
    :param region:
    :param lines: lines == number of lines on the region
    :return:
    """
    global counter
    mask = img.copy()
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), theta = rect
    box = cv2.boxPoints(((x, y), (w, h), theta))
    box = np.int0(box)
    # midPt == the front of the car
    midPt = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
    midPt2 = ((box[2][0] + box[3][0]) // 2, (box[2][1] + box[3][1]) // 2)
    car_w = np.array([box[0][0] - box[1][0]])
    car_w = np.linalg.norm(car_w)
    car_h = np.array([box[1][1] - box[2][1]])
    car_h = np.linalg.norm(car_h)
    ratio = round(car_h / car_w, 2)
    ratio2 = round(w / h, 2)
    # ensures the car is inside the roi corner
    path = mplPath.Path(roi_corners)

    # finish line
    x0, y0 = roi_corners[0][0], roi_corners[0][1]
    x1, y1 = roi_corners[3][0], roi_corners[3][1]
    if path.contains_point(midPt):
        cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)
        cv2.circle(mask, midPt, 5, (0, 255, 255), -1)
        cv2.line(mask, midPt2, midPt, (255, 0, 0), 3)
        cv2.putText(mask, str(ratio), (box[2][0] + 10, box[2][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        cv2.putText(mask, str(ratio2), (box[2][0] + 100, box[2][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        if region:
            # draw regions
            boxLength = abs(roi_corners[1][1] - roi_corners[0][1])
            num_regions = lines + 1
            # splits into multiple region
            for step in np.arange(boxLength // num_regions + 1, boxLength, boxLength // num_regions):
                region_pts = find_equal_line(roi_corners, roi_corners[1][1] + step - 20)
                x_left, y_left, x_right, y_right = region_pts.reshape(4)
                # lines to separate multiple regions
                effs = np.polyfit((x_left, x_right), (y_left, y_right), 1)
                poly = np.poly1d(effs)
                # when car intercepts region lines, then report
                if abs(midPt[1] - poly(midPt[0])) <= 5:
                    cv2.line(mask, (x_left, y_left), (x_right, y_right), (0, 0, 255), 3)
                    counter += 1
                    print(counter)

        else:
            effs = np.polyfit((x0, x1), (y0, y1), 1)
            poly = np.poly1d(effs)
            if abs(midPt[1] - poly(midPt[0])) <= 10:
                cv2.line(mask, (x0, y0), (x1, y1), (0, 0, 255), 4)
                counter += 1
                print(counter)
    return mask


cap = cv2.VideoCapture('car.mp4')
_, frame = cap.read()
H = frame.shape[0]
W = frame.shape[1]
backSub = cv2.createBackgroundSubtractorMOG2(500, 50, True)
frame = cv2.resize(frame, (W // 2, H // 2))
pts = []
avg = []
w_avg = []
h_avg = []
roi_corners3 = [[0, 305], [165, 90], [255, 120], [205, frame.shape[0]]]
counter = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (W // 2, H // 2))
    lineImg = frame.copy()

    roi = frame[124:240, 49:233]
    fgMask = backSub.apply(frame)
    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    fgMask = filter_mask(fgMask)

    canny = cv2.Canny(fgMask, 40, 180)
    lineMask = fgMask.copy()
    lineMask = draw_lines(fgMask, 62)
    cntImg = fgMask.copy()
    cntImg = cv2.cvtColor(cntImg, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(fgMask)
    cntImg2 = fgMask.copy()
    cntImg2 = cv2.cvtColor(cntImg2, cv2.COLOR_GRAY2BGR)
    cntImg3 = fgMask.copy()
    cntImg3 = cv2.cvtColor(cntImg3, cv2.COLOR_GRAY2BGR)
    # draw roi and region
    lineImg = draw_roi(lineImg, roi_corners3, True, 2)
    trackImg = lineImg.copy()
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    center = None

    for (i, contour) in enumerate(contours):
        epsilon = 0.00001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        (x, y, w, h) = cv2.boundingRect(contour)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            contour_valid = (w >= 25) and (
                    h >= 25)
            if not contour_valid:
                continue
            else:
                if center is not None:
                    # draw rotated rects
                    rect = cv2.minAreaRect(contour)
                    lineMask = draw_lines(fgMask, 62, rect)
                    angle = rect[2]
                    if not math.isnan(angle) and angle != 0 and 40 < abs(angle) < 80:
                        # draw straight line through the contours
                        cv2.drawContours(cntImg3, contour, -1, (0, 0, 255), 3)
                        rows, cols = cntImg2.shape[:2]
                        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        lefty = int((-x * vy / vx) + y)
                        righty = int(((cols - x) * vy / vx) + y)
                        cv2.line(cntImg2, (cols - 1, righty), (0, lefty), (0, 255, 0), 3)
                        # draw rect on contour
                        (x, y), (w, h), theta = rect
                        box = cv2.boxPoints(((x, y), (w, h), theta))
                        box = np.int0(box)
                        cv2.drawContours(cntImg, [box], 0, (0, 0, 255), 2)
                        trackImg = draw_roi_track(trackImg, roi_corners3, contour, True, 2)

                        avg.append(angle)
                        w_avg.append(w)
                        h_avg.append(h)
                    pts.append(center)

    cv2.imshow('frame', frame)
    cv2.imshow('fgMask', fgMask)
    cv2.imshow('cntImg', cntImg)
    cv2.imshow('cntImg2', cntImg2)
    cv2.imshow('cntImg3', cntImg3)

    cv2.imshow('lineMask', lineMask)
    cv2.imshow('lineImage', lineImg)
    cv2.imshow('trackImg', trackImg)

    k = cv2.waitKey(50) & 0xff
    if k == ord('q'):
        break
    if k == ord('p'):
        k == cv2.waitKey(0)

# record theta and the size of the vehicles
array = np.array(avg)
w_avg = np.array(w_avg)
h_avg = np.array(h_avg)
print(int(np.mean(array)))
print(int(np.mean(w_avg)))
print(int(np.mean(h_avg)))
cap.release()
cv2.destroyAllWindows()
