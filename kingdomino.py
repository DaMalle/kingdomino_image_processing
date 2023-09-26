import cv2 as cv


def main() -> None:
    img = cv.imread("1.jpg")
    for i in range(0, 5):
        for j in range(0, 5):
            current_img = img[100*i:100*(i+1), 100*j:100*(j+1)]
            cv.imshow(f"test{i}{j}.jpg", current_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
