import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ratio.png', 0)
headlines = img[295:323, 285:345]
cv2.imshow('car', headlines)
cv2.imwrite('headlines.jpg', headlines)
# grey = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
w, h = headlines.shape[::-1]
res = cv2.matchTemplate(img, headlines, cv2.TM_CCOEFF_NORMED)
threshold = 0.51
loc = np.where(res >= threshold)
print(loc)
p2 = 0
for pt in zip(*loc[::-1]):
    if abs(pt[0] - p2) >= 100:
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        p2 = pt[0]


cv2.imshow('image', img)

plt.imshow(img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
