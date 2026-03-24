from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import maze_loader as ml

#Uses BFS to find shortest path to exit
def bfs(maze, start, end):

    rows, cols = maze.shape
    queue = deque([start])
    parent = {start: None}
    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr, nc] == 0 and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
    #recreate the path
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    return path[::-1]  # reverse

def solve_maze(filename):
    maze, original_img = ml.loadmaze(filename)
    rows, cols = maze.shape
    #find entrance and exit
    start_col = np.where(maze[0] == 0)[0][len(np.where(maze[0] == 0)[0]) // 2]
    end_col = np.where(maze[-1] == 0)[0][len(np.where(maze[-1] == 0)[0]) // 2]
    start = (0, int(start_col)) #top
    end = (rows - 1, int(end_col)) #bottom
    print(f"Start: {start}, End: {end}")
    path = bfs(maze, start, end)
    print(f"Path length: {len(path)} pixels")
    # Draw solution
    img_color = original_img.convert("RGB")
    draw = ImageDraw.Draw(img_color)
    path_xy = [(c, r) for r, c in path]
    draw.line(path_xy, fill=(0, 0, 255), width=4)
    # Save output
    output_path = filename.replace(".png", "_bfs.png")
    img_color.save(output_path)
    print(f"Saved to {output_path}")
    
    return maze

if __name__ == "__main__":
    maze = solve_maze("MAZE_0.png")
    
    # MAZE_1
    ml.loadHazardsMaze("MAZE_1.png")
    