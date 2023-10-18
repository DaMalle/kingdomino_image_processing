import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum


class Terrain(Enum):
    FOREST = 1  # green with trees
    PLAINS = 2  # yellow
    GRASSLANDS = 3  # light green
    WASTELANDS = 4  # muddy gray/brown
    OCEAN = 5  # blue
    MINE = 6  # black
    UNDEFINED = 0  # other


board = np.full(shape=(7, 7), fill_value=-1)
unvisitied = [(x, y) for y in range(1, 6) for x in range(1, 6)]
visited = []
connected_components = []

# test
board = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, Terrain.PLAINS, Terrain.MINE, Terrain.MINE, Terrain.MINE, Terrain.UNDEFINED, -1],
        [-1, Terrain.OCEAN, Terrain.FOREST, Terrain.WASTELANDS, Terrain.WASTELANDS, Terrain.UNDEFINED, -1],
        [-1, Terrain.OCEAN, Terrain.OCEAN, Terrain.UNDEFINED, Terrain.WASTELANDS, Terrain.PLAINS, -1],
        [-1, Terrain.OCEAN, Terrain.OCEAN, Terrain.FOREST, Terrain.PLAINS, Terrain.PLAINS, -1],
        [-1, Terrain.OCEAN, Terrain.OCEAN, Terrain.FOREST, Terrain.FOREST, Terrain.FOREST, -1],
        [-1, -1, -1, -1, -1, -1, -1]
])


def plot_histogram(img) -> None:
    plt.plot(cv.calcHist([img], [0], None, [180], [0, 180]))
    plt.xlim([0, 180])
    plt.show()


def main() -> None:
    raw_image = cv.imread("14.jpg")
    hsv_image = cv.cvtColor(raw_image, cv.COLOR_BGR2HSV)
    lower_sand = np.array([0, 180, 0])
    upper_sand = np.array([179, 255, 255])
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    for i in range(0, 5):
        for j in range(0, 5):
            print(f"j: {j}, i: {i}")
            cell = hsv_image[100*i:100*(i+1), 100*j:100*(j+1)]
            mask = cv.inRange(cell, lower_blue, upper_blue)
            # mask = cv.inRange(cell, lower_sand, upper_sand)
            cv.imshow("mask.jpg", mask)
            cv.imshow(f"test{i}{j}.jpg", cell)
            cv.waitKey(0)
            cv.destroyAllWindows()
    cv.waitKey(0)
    cv.destroyAllWindows()


def visit_node(node: tuple[int, int], component) -> None:
    if node in visited:
        return
    x, y = node
    if component is None:
        component = []
        connected_components.append(component)
        current_value = board[y][x]
    else:
        x0, y0 = component[0]
        current_value = board[y0][x0]
    if board[y][x] == current_value:
        component.append((x, y))
        visited.append(node)
        unvisitied.remove((x, y))
        visit_node((x, y-1), component)
        visit_node((x, y+1), component)
        visit_node((x-1, y), component)
        visit_node((x+1, y), component)
    return


if __name__ == "__main__":
    # main()

    while unvisitied:
        visit_node(unvisitied[0], None)
    print(connected_components)
