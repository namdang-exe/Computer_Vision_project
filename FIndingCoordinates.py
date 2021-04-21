import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("car.jpg", 0)

car = img[125:220, 200:300]  # roi of the image

indices = np.where(car != [0])
coordinates = zip(indices[0], indices[1])

# cv2.imshow('coordinates', coordinates)
plt.imshow(img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()