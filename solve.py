from PIL import Image
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def load_maze_from_png(filename, threshold=128):
    img = Image.open(filename).convert("L")
    maze_array = np.array(img)
    maze = (maze_array < threshold).astype(int)
    return maze


def solve_maze(maze, start, end):
    rows, cols = maze.shape
    visited = np.zeros_like(maze)
    prev = np.full((rows, cols, 2), -1)

    q = deque([start])
    visited[start] = 1

    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while q:
        r, c = q.popleft()

        if (r, c) == end:
            break

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and not visited[nr, nc]:
                q.append((nr, nc))
                visited[nr, nc] = 1
                prev[nr, nc] = (r, c)

    path = []
    r, c = end

    while prev[r, c][0] != -1:
        path.append((r, c))
        r, c = prev[r, c]

    path.append(start)
    path.reverse()

    return path


def visualize_maze(maze, path=None):
    plt.figure(figsize=(10,10))
    plt.imshow(maze, cmap="gray")

    if path:
        pr, pc = zip(*path)
        plt.plot(pc, pr, color="red", linewidth=2)

    plt.axis("off")
    plt.show()


# -------- RUN THE PROGRAM --------

maze = load_maze_from_png("MAZE_0.png")

start = (1,1)
end = (maze.shape[0]-2, maze.shape[1]-2)

path = solve_maze(maze, start, end)

visualize_maze(maze, path)
