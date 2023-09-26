import cv2 as cv
import numpy as np


def divide_img_blocks(img, number_blocks=(2, 2)):
    horizontal = np.array_split(img, number_blocks[0])
    splitted_img = [np.array_split(block, number_blocks[1], axis=1) for block in horizontal]
    return np.asarray(splitted_img, dtype=np.ndarray).reshape(number_blocks)


def main() -> None:
    img = cv.imread("1.jpg")
    height, width, channels = img.shape
    # img = img[height//2, :width//2:]
    # img = divide_img_blocks(img)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         cv.imwrite("test", img[i,j])
    # cv.imshow("test", img)
    left = img[:height//5, :width//5]
    cv.imshow("test", left)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
