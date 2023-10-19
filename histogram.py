import matplotlib.pyplot as plt
import cv2 as cv


def image_historgram(image):
    raw_image = cv.imread(image)
    hsv_image = cv.cvtColor(raw_image, cv.COLOR_BGR2HSV)
    i, j = 2, 4
    cell = hsv_image[100*i:100*(i+1), 100*j:100*(j+1)]
    raw_cell = raw_image[100*i:100*(i+1), 100*j:100*(j+1)]
    # cv.imshow("cell.jpg", cell)
    # cv.imshow("raw.jpg", raw_cell)

    histogram_1 = cv.calcHist([cell], [0], None, [180], [0, 180])
    plt.plot(histogram_1, color='r')
    plt.xlim([0, 180])

    histogram_2 = cv.calcHist([cell], [1], None, [256], [0, 256])
    plt.plot(histogram_2, color='g')
    plt.xlim([0, 256])
    histogram_3 = cv.calcHist([cell], [2], None, [256], [0, 256])
    plt.plot(histogram_3, color='b')
    plt.xlim([0, 256])
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    image_historgram("14.jpg")
