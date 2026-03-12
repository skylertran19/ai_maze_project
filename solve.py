from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ---------------- LOAD MAZE ----------------
def load_maze(filename):
    img = Image.open(filename).convert("L")
    maze_array = np.array(img)
    
    # Auto-detect path vs wall
    if np.mean(maze_array) > 128:
        maze = (maze_array < 128).astype(int)  # 1 = wall, 0 = free
    else:
        maze = (maze_array > 128).astype(int)
    return maze

# ---------------- BFS SOLVER ----------------
def bfs_solve(maze, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {}
    moves = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

    while queue:
        current = queue.popleft()
        if current == goal:
            # Reconstruct path
            path = []
            while current != start:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1]  # start -> goal

        r, c = current
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
                if maze[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = current
                    queue.append((nr, nc))
    return None  # no path found

# ---------------- VISUALIZE ----------------
def visualize(maze, path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap="gray")
    if path:
        pr, pc = zip(*path)
        plt.plot(pc, pr, linewidth=2, color='red')
    plt.axis("off")
    plt.title("Maze Solution (BFS)" if path else "Maze")
    plt.show()

# ---------------- MAIN ----------------
maze = load_maze("MAZE_0.png")

# ---------------- AUTO-SELECT START AND GOAL ----------------
rows, cols = maze.shape
print(f"Maze size: {rows} x {cols}")

# Start: first free cell from top-left
start = None
for r in range(rows):
    for c in range(cols):
        if maze[r, c] == 0:
            start = (r, c)
            break
    if start:
        break

# Goal: first free cell from bottom-right
goal = None
for r in reversed(range(rows)):
    for c in reversed(range(cols)):
        if maze[r, c] == 0:
            goal = (r, c)
            break
    if goal:
        break

print(f"Start set to: {start}, Goal set to: {goal}")

# ---------------- SOLVE ----------------
solution_path = bfs_solve(maze, start, goal)
if solution_path is None:
    print("No path found. Check maze or start/goal positions.")
else:
    print("Solution path length:", len(solution_path))
    visualize(maze, solution_path)