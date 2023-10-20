import cv2 as cv
import numpy as np
from enum import Enum
from crown_detection import get_crowns

# Defining the class Terrain with enumerations for the different terrain types
class Terrain(Enum):
    FOREST = 1      # green with trees
    PLAIN = 2       # light green
    SAVANNAH = 3    # yellow
    WASTELAND = 4   # muddy gray/brown
    OCEAN = 5       # blue
    MINE = 6        # black
    UNDEFINED = 0   # other

# Creates numpy arrays to be filled with values later on, one for terrain type, and one for amount of crowns on a field
board = np.full(shape=(5, 5), fill_value=Terrain.UNDEFINED)
crowns = np.full(shape=(5, 5), fill_value=0)
# Array for all fields on the board before being run through the terrain algorithm
unvisitied = [(x, y) for y in range(5) for x in range(5)]
# Array for the visited fields
visited = []
# 2D array for connected fields
connected_terrains = []

# Defining upper and lower values for thresholding the different terrain types
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
    lower = np.array([10, 150, 100])
    upper = np.array([55, 240, 180])
    mask = cv.inRange(cell, lower, upper)
    return cv.countNonZero(mask)


def match_wasteland(cell) -> None:
    lower = np.array([20, 50, 40])
    upper = np.array([40, 256, 130])
    mask = cv.inRange(cell, lower, upper)
    return cv.countNonZero(mask)


def match_mine(cell) -> int:
    lower = np.array([0, 0, 0])
    upper = np.array([175, 140, 50])
    mask = cv.inRange(cell, lower, upper)
    return cv.countNonZero(mask)


def get_best_terrain_match(cell):
    # Sets names for the different terrain types with their functions
    terrains = {
        "OCEAN": match_ocean(cell),
        "MINE": match_mine(cell),
        "WASTELAND": match_wasteland(cell),
        "PLAIN": match_plain(cell),
        "SAVANNAH": match_savanna(cell),
        "FOREST": match_forest(cell)
    }
    key = max(terrains, key=terrains.get)
    # If the amount of thresholded pixels never exceed 2000 in any case make the terrain undefined
    if terrains[key] < 2000:
        key = "UNDEFINED"

    # Go through each case for terrain
    match key:
        case "OCEAN": return Terrain.OCEAN
        case "MINE": return Terrain.MINE
        case "SAVANNAH": return Terrain.SAVANNAH
        case "WASTELAND": return Terrain.WASTELAND
        case "PLAIN": return Terrain.PLAIN
        case "FOREST": return Terrain.FOREST
        case _: return Terrain.UNDEFINED

# Doesn't actually run though
def main() -> None:
    # Create the first image and convert it to HSV
    raw_image = cv.imread("1.jpg")
    hsv_image = cv.cvtColor(raw_image, cv.COLOR_BGR2HSV)
    # Go through the image and split it into cells
    for i in range(0, 1):
        for j in range(0, 1):
            print(f"j: {j}, i: {i}")
            cell = hsv_image[100*i:100*(i+1), 100*j:100*(j+1)]
            # cv.imshow("hsv.jpg", hsv_image)
            # Print and show the cell
            print(get_best_terrain_match(cell))
            cv.imshow(f"test{i}{j}.jpg", cell)
            cv.waitKey(0)
            cv.destroyAllWindows()
    cv.waitKey(0)
    cv.destroyAllWindows()

# Terrain assignment function
def fill_board_terrains(image) -> None:
    # Changes the image to use the HSV channels
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Goes through all the cells in the image rows and columns
    for i in range(0, 5):
        for j in range(0, 5):
            cell = hsv_image[100*i:100*(i+1), 100*j:100*(j+1)]
            # Inserts the best terrain match into the board array
            board[i][j] = get_best_terrain_match(cell)

# Defines the points function
def print_board_points() -> None:
    # Defines sum as 0 to begin with
    sum = 0
    # Goes through all the cells in the connected terrains array
    for connected_terrain in connected_terrains:
        # Sets the starting crown count as 0
        crown_count = 0
        # Checks the amount of crowns in each cell and the corresponding terrains that are connected
        for crown in [crowns[y][x] for x, y in connected_terrain]:
            # Adds the amount of crowns to the crown count
            crown_count += crown
        # Calculates the amount of points with the amount of cells times the amount of crowns in those cells
        sum += (len(connected_terrain) * crown_count)
    # Prints the sum of points
    print(f"Total points: {sum}")

# The cell visiting function
def visit_node(node, grouped_terrain, current_terrain) -> None:
    # Checks if grouped_terrain is None, which it is
    if grouped_terrain is None:
        # Creates an empty array to append connected terrain into
        grouped_terrain = []
        # Appends the connected terrains in a 2D array to collect all terrains
        connected_terrains.append(grouped_terrain)
    # Splits the x, y values from node
    x, y = node
    # Checks if the current terrain is the same type as the last one
    if board[y][x] == current_terrain:
        # Appends the x, y values from the current cell to the array for grouping current terrain
        grouped_terrain.append((x, y))
        # Appends cell coordinates to the visited array and removes them from the unvisited array
        visited.append(node)
        unvisitied.remove((x, y))
        # Goes through all the adjacent cells
        for x0, y0 in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]:
            # Checks if there is overflow and if the cell has not been visited yet
            if 0 <= y0 < 5 and 0 <= x0 < 5 and (x0, y0) not in visited:
                # Goes through visit_node with the adjacent cell
                visit_node((x0, y0), grouped_terrain, current_terrain)
    return

# Main
if __name__ == "__main__":
    # main()
    # Sets the raw_image as a given board image
    raw_image = cv.imread("14.jpg")
    # Runs the terrain assigning function with the given image
    fill_board_terrains(raw_image)
    # Finds the amount of crowns in given image
    get_crowns(crowns, raw_image)
    # While there are still unvisited cells, continue visiting the rest
    while unvisitied:
        x, y = unvisitied[0]
        visit_node((x, y), None, board[y][x])
    # Print the final board score
    print_board_points()