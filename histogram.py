import matplotlib.pyplot as plt
import cv2 as cv


def image_historgram(image):
    # Defines the raw image
    raw_image = cv.imread(image)
    # Converts BGR image to HSV
    hsv_image = cv.cvtColor(raw_image, cv.COLOR_BGR2HSV)
    # sets coordinates for the top left corner of the image
    i, j = 0, 0
    # creates the cell of the i, j coordinates
    cell = hsv_image[100*i:100*(i+1), 100*j:100*(j+1)]
    raw_cell = raw_image[100*i:100*(i+1), 100*j:100*(j+1)]
    # cv.imshow("cell.jpg", cell)
    # cv.imshow("raw.jpg", raw_cell)

    # Makes a histogram for the Hue value
    histogram_1 = cv.calcHist([cell], [0], None, [180], [0, 180])
    plt.plot(histogram_1, color='r')
    plt.xlim([0, 180])

    # Makes a histogram for the Saturation value
    histogram_2 = cv.calcHist([cell], [1], None, [256], [0, 256])
    plt.plot(histogram_2, color='g')
    plt.xlim([0, 256])

    # Makes a histogram for the Value value
    histogram_3 = cv.calcHist([cell], [2], None, [256], [0, 256])
    plt.plot(histogram_3, color='b')
    plt.xlim([0, 256])

    # Shows the histograms
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    image_historgram("1.jpg")
