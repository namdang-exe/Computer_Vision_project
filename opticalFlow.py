import cv2 as cv
import numpy as np

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=1, blockSize=1)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("car.mp4")
# Variable for color to draw optical flow track
color = (0, 255, 0)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
prev_point = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
mask = np.zeros_like(first_frame)
pts = []
(dx, dy) = 0, 0
while cap.isOpened():
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates sparse optical flow by Lucas-Kanade method
    next_point, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_point, None, **lk_params)
    # Selects good feature points from previous position
    good_old = prev_point[status == 1]
    # Selects good feature points for next position
    good_new = next_point[status == 1]
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        x1, y1 = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        x0, y0 = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv.line(mask, (x0, y0), (x1, y1), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        # frame = cv.circle(frame, (x1, y1), 3, color, -1)
        pts.append((x1, y1))

    array = np.array(pts)
    array = np.int0(array)

    for i in range(len(array)):
        if array[i - 1] is None or array[i] is None:
            continue
        else:
            cv.circle(mask, (array[i][0], array[i][1]), 3, (255, 0, 0), -1)
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev_point = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    cv.imshow("sparse optical flow", output)
    cv.imshow("frame", frame)
    cv.imshow("Mask", mask)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(40) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
