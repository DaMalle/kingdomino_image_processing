import cv2 as cv
import numpy as np


def get_matched_locations(raw_image, template):
    result = cv.matchTemplate(raw_image, template, cv.TM_CCOEFF_NORMED)
    locations = list(zip(*np.where(result >= 0.6)[::-1]))
    return locations


gaussian_filter = (1/273) * np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
])


def get_crowns(crown_grid, image):
    raw_image = cv.imread(image)
    # cv.imshow("raw.jpg", raw_image)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    raw_image = cv.filter2D(raw_image, -1, gaussian_filter)
    sharpened_image = cv.filter2D(raw_image, -1, kernel)
    # sharpened_image = cv.filter2D(sharpened_image, -1, kernel)
    # sharpened_image = cv.filter2D(sharpened_image, -1, gaussian_filter)
    # cv.imshow("sharpened.jpg", sharpened_image)

    template = cv.imread("crown.jpg")
    # template = cv.filter2D(template, -1, gaussian_filter)
    # template = cv.filter2D(template, -1, kernel)
    # template = cv.filter2D(template, -1, kernel)
    # template = cv.filter2D(template, -1, gaussian_filter)
    templates = [template]
    templates.append(cv.rotate(templates[0], cv.ROTATE_90_CLOCKWISE))
    templates.append(cv.rotate(templates[1], cv.ROTATE_90_CLOCKWISE))
    templates.append(cv.rotate(templates[2], cv.ROTATE_90_CLOCKWISE))
    locations = []
    for template in templates:
        locations += get_matched_locations(sharpened_image, template)

    print(len(locations))
    needle_width = templates[0].shape[1]
    needle_height = templates[0].shape[0]
    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    rectangles = [
        (location[0], location[1], needle_width, needle_height)
        for location in locations
    ]

    rectangles, _ = cv.groupRectangles(rectangles, 1, 0.5)
    print(len(rectangles))
    for x, y, w, h in rectangles:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv.rectangle(raw_image, top_left, bottom_right, line_color, line_type)
        crown_grid[y//100][x//100] += 1
    cv.imshow("crowns.png", raw_image)
    print(crown_grid)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # get_crowns()
    pass
