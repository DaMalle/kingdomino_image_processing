import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def main():
    raw_img = cv.imread("1.jpg")
    gray_img = raw_img # cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    template = gray_img[305:340, 405:430] #[315:330, 415:422]  # [305:340, 405:430]
    cv.imshow("test.jpg", template)
    test_img = cv.matchTemplate(gray_img, template, cv.TM_CCOEFF)

    cv.imshow("test.jpg", test_img)
    hsv_img = cv.cvtColor(raw_img, cv.COLOR_BGR2HSV)

    # Thresholding
    lower = np.array([25, 50, 145], np.uint8)
    upper = np.array([50, 100, 190], np.uint8)
    thresholding_img = cv.inRange(hsv_img, lower, upper)
    # cv.imshow("test.jpg", thresholding_img)
    # plt.plot(cv.calcHist([hsv], [0], None, [180], [0, 180]))
    # plt.xlim([0, 180])
    # plt.plot(cv.calcHist([hsv], [1], None, [255], [0, 255]))
    # plt.xlim([0, 255])
    # plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
