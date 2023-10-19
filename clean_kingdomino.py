import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
from sharp_crown_detection import get_crowns


class Terrain(Enum):
    FOREST = 1      # green with trees
    PLAIN = 2      # light green
    SAVANNAH = 3    # yellow
    WASTELAND = 4  # muddy gray/brown
    OCEAN = 5       # blue
    MINE = 6        # black
    UNDEFINED = 0   # other


board = np.full(shape=(5, 5), fill_value=-1)
crowns = np.full(shape=(5, 5), fill_value=0)
unvisitied = [(x, y) for y in range(5) for x in range(5)]
visited = []
connected_terrains = []


def match_forest(cell) -> None:
    lower = np.array([30, 100, 0])
    upper = np.array([80, 200, 80])
    mask = cv.inRange(cell, lower, upper)
    return cv.countNonZero(mask)


def match_ocean(cell) -> None:
    lower = np.array([100, 240, 140])
    upper = np.array([120, 256, 180])
    mask = cv.inRange(cell, lower, upper)
    return cv.countNonZero(mask)


def match_savanna(cell) -> None:
    lower = np.array([20, 240, 160])
    upper = np.array([40, 256, 190])
    mask = cv.inRange(cell, lower, upper)
    return cv.countNonZero(mask)


def match_plain(cell) -> None:
    lower = np.array([180, 256, 256])
    upper = np.array([180, 256, 256])
    mask = cv.inRange(cell, lower, upper)
    # cv.imshow("mask.jpg", mask)
    # print(cv.countNonZero(mask))
    return cv.countNonZero(mask)


def match_wasteland(cell) -> None:
    lower = np.array([180, 256, 256])
    upper = np.array([180, 256, 256])
    mask = cv.inRange(cell, lower, upper)
    # cv.imshow("mask.jpg", mask)
    return cv.countNonZero(mask)


def match_mine(cell) -> int:
    lower = np.array([180, 256, 256])
    upper = np.array([180, 256, 256])
    mask = cv.inRange(cell, lower, upper)
    # cv.imshow("mask.jpg", mask)
    return cv.countNonZero(mask)


def get_best_terrain_match(cell):
    terrains = {
        "OCEAN": match_ocean(cell),
        "MINE": match_mine(cell),
        "WASTELAND": match_wasteland(cell),
        "PLAIN": match_plain(cell),
        "SAVANNAH": match_savanna(cell),
        "FOREST": match_forest(cell)
    }
    key = terrains.get
    max_key = max(terrains, key=key)

    if terrains[max_key] < 4000:
        max_key = "UNDEFINED"

    match max_key:
        case "OCEAN": return Terrain.OCEAN
        case "MINE": return Terrain.MINE
        case "SAVANNAH": return Terrain.SAVANNAH
        case "WASTELAND": return Terrain.WASTELAND
        case "PLAIN": return Terrain.PLAIN
        case "FOREST": return Terrain.FOREST
        case _: return Terrain.UNDEFINED


def main() -> None:
    raw_image = cv.imread("14.jpg")
    hsv_image = cv.cvtColor(raw_image, cv.COLOR_BGR2HSV)
    for i in range(0, 1):
        for j in range(0, 1):
            print(f"j: {j}, i: {i}")
            cell = hsv_image[100*i:100*(i+1), 100*j:100*(j+1)]
            cv.imshow("hsv.jpg", hsv_image)
            # match_forest(cell)
            # match_ocean(cell)
            # match_savanna(cell)
            # match_plain(cell)
            print(get_best_terrain_match(cell))
            cv.imshow(f"test{i}{j}.jpg", cell)
            cv.waitKey(0)
            cv.destroyAllWindows()
    cv.waitKey(0)
    cv.destroyAllWindows()


def print_board_points() -> None:
    sum = 0
    for connected_terrain in connected_terrains:
        crown_count = 0
        for crown in [crowns[y][x] for x, y in connected_terrain]:
            crown_count += crown
        sum += (len(connected_terrain) * crown_count)
    print(f"Total points: {sum}")


def visit_node(node, grouped_terrain, current_terrain) -> None:
    if grouped_terrain is None:
        grouped_terrain = []
        connected_terrains.append(grouped_terrain)
    x, y = node
    if board[y][x] == current_terrain:
        grouped_terrain.append((x, y))
        visited.append(node)
        unvisitied.remove((x, y))
        for x0, y0 in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]:
            if 0 <= y0 < 5 and 0 <= x0 < 5 and (x0, y0) not in visited:
                visit_node((x0, y0), grouped_terrain, current_terrain)
    return


if __name__ == "__main__":
    main()
    # # use cv.countNonZero(img)
    # get_crowns(crowns, "14.jpg")
    # while unvisitied:
    #     x, y = unvisitied[0]
    #     visit_node((x, y), None, board[y][x])
    # print(connected_terrains)
    # print_board_points()