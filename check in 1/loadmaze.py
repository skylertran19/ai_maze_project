from PIL import Image
import numpy as np
import csv
"""
Matrix code(no access to agent)
0 Empty/navigable cell
1 Wall
2 Start 'S'
3 Goal/Exit/Target 'G'
4 Death pits(instant death, respawn at S) 'P'
5 Teleport pad(to another location) 'T'       ---destination is deterministic for each specific teleport pad, teleport destinations are navigable calls
? Confusion pads(inverts agent control) 'C'
"""

def load_maze_from_png(filename, output_csv=None):
    # Maze expected to have 16px per cell (2px walls + 14px passages)
    """
    Returns a 64x64 numpy array where each cell stores a bitmask of its walls:
        bit 0 (1) = wall to the RIGHT
        bit 1 (2) = wall BELOW
    
    So cell value 0 = open on right and bottom
       cell value 1 = wall on right only
       cell value 2 = wall below only
       cell value 3 = wall on both right and bottom
    
    Use can_move_right(maze, r, c) and can_move_down(maze, r, c) to check moves.
    """
    img = Image.open(filename).convert('L')
    arr = np.array(img)

    step   = 16
    n      = 64
    maze   = np.zeros((n, n), dtype=int)

    for r in range(n):
        for c in range(n):
            # wall to the right of this cell?
            if c < n - 1:
                px_r = r * step + 8
                px_c = (c + 1) * step
                if arr[px_r, px_c] < 128:
                    maze[r, c] |= 1   # set bit 0

            # wall below this cell?
            if r < n - 1:
                px_r = (r + 1) * step
                px_c = c * step + 8
                if arr[px_r, px_c] < 128:
                    maze[r, c] |= 2   # set bit 1

    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in maze:
                writer.writerow(row)

    return maze


def can_move_right(maze, r, c):
    return not (maze[r, c] & 1)

def can_move_down(maze, r, c):
    return not (maze[r, c] & 2)

def can_move_left(maze, r, c):
    return c > 0 and not (maze[r, c-1] & 1)

def can_move_up(maze, r, c):
    return r > 0 and not (maze[r-1, c] & 2)
    #teleports:Green circle(🟢) to Green Star(✳️), Yellow Circle(🟡) to Yellow Star and Purple Circle(🟣)  to Purple Star

def visualize_maze(maze, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # 0 = open (white), 1 = wall (black), 2 = path (red)
    cmap = ListedColormap(['white', 'black', 'red'])

    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

def fakeSolution(output_csv):
    #used claude to generate a simple "maze" to test proper visualization
    size = 64
    grid = np.zeros((size, size), dtype=int)

    # # Outer border
    # grid[0, :] = 1
    # grid[-1, :] = 1
    # grid[:, 0] = 1
    # grid[:, -1] = 1

    # Some internal walls
    grid[2, :12] = 1
    grid[5, 4:] = 1
    grid[:8, 8] = 1
    grid[10:, 3] = 1
    grid[20, 20:50] = 1
    grid[20:40, 50] = 1
    grid[40, 10:50] = 1
    # Fake path: down col 0, then right across last row
    solution = [(r, 0) for r in range(64)] + [(63, c) for c in range(1, 64)]
    for (r, c) in solution:
        grid[r][c] = 2

    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for r in grid:
                writer.writerow(r)

    return grid.tolist()

maze = load_maze_from_png("MAZE_0.png", output_csv="csvMaze0.csv")
# maze = maze_to_cells(maze)
visualize_maze(maze, save_path="visMaze0.png")

maze = load_maze_from_png("MAZE_1.png", output_csv="csvMaze1.csv")
visualize_maze(maze, save_path="vizMaze1.png")

fake = fakeSolution(output_csv="csvFakeSolution.csv")
visualize_maze(fake, save_path='FAKE.png')
 