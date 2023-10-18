import cv2 as cv
import numpy as np


def get_template_locations(raw_image, template):
    result = cv.matchTemplate(raw_image, template, cv.TM_CCOEFF_NORMED)
    locations = list(zip(*np.where(result >= 0.6)[::-1]))
    return locations


if __name__ == "__main__":
    raw_image = cv.imread("2.jpg")
    templates = [cv.imread("crown.jpg")]
    templates.append(cv.rotate(templates[0], cv.ROTATE_90_CLOCKWISE))
    templates.append(cv.rotate(templates[1], cv.ROTATE_90_CLOCKWISE))
    templates.append(cv.rotate(templates[2], cv.ROTATE_90_CLOCKWISE))
    locations = []
    for template in templates:
        locations += get_template_locations(raw_image, template)

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
    cv.imshow("test.png", raw_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
