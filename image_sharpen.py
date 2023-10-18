import cv2 as cv
import numpy as np


raw_image = cv.imread("14.jpg")
kernel1 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
kernel = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])
kernel2 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])
# gaussian_blur_image = cv.filter2D(raw_image, -1, kernel/16)
raw_image = cv.medianBlur(raw_image, 3)
sharpened_image = cv.filter2D(raw_image, -1, kernel2)
# sharpened_image = cv.filter2D(raw_image, -1, kernel2)
# gaussian_blur_image = cv.filter2D(sharpened_image, -1, kernel1)
cv.imshow("raw.jpg", raw_image)
cv.imshow("sharpened.jpg", sharpened_image)
# cv.imshow("gassian_blur.jpg", gaussian_blur_image)

cv.waitKey(0)
cv.destroyAllWindows()
