import cv2 as cv
import numpy as np

# Function for checking which locations in a given image include a crown
def get_matched_locations(raw_image, template):
    result = cv.matchTemplate(raw_image, template, cv.TM_CCOEFF_NORMED)
    locations = list(zip(*np.where(result >= 0.6)[::-1]))
    return locations

# Creates a 2D array housing a gaussian filter
gaussian_filter = (1/273) * np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
])

# Are we even using this?!?!??
# Why is this still here D:
def get_print_crowns(crown_grid, image):
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

# Crown finding function
def get_crowns(crown_grid, image):
    # Defines a kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    # Blurs the image with the gaussian filter
    blur_image = cv.filter2D(image, -1, gaussian_filter)
    # Sharpens the blurred image with the kernel
    sharpened_image = cv.filter2D(blur_image, -1, kernel)

    # Defines the template as the image of the crown
    template = cv.imread("crown.jpg")
    # Creates an array with the crown image but rotated 90 degrees a few times to get one for each rotation
    templates = [template]
    templates.append(cv.rotate(templates[0], cv.ROTATE_90_CLOCKWISE))
    templates.append(cv.rotate(templates[1], cv.ROTATE_90_CLOCKWISE))
    templates.append(cv.rotate(templates[2], cv.ROTATE_90_CLOCKWISE))

    # Creates array for crown locations
    locations = []
    # For each rotation of the template it will check for crowns
    for template in templates:
        # Counts the amount of locations a crown is found
        locations += get_matched_locations(sharpened_image, template)

    # Gets the width of the first template where the crown is horizontal
    needle_width = templates[0].shape[1]
    # And the height of the second template in which the crown is vertical to get a square
    needle_height = templates[0].shape[0]

    # Goes through the list of possible crowns and groups the possible locations into squares
    rectangles = [
        (location[0], location[1], needle_width, needle_height)
        for location in locations
    ]

    rectangles, _ = cv.groupRectangles(rectangles, 1, 0.5)
    for x, y, w, h in rectangles:
        # Counts the amount of crowns in the individual cells
        crown_grid[y//100][x//100] += 1
