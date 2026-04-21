from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import maze_loader as ml
# import maze_environment as me

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

class testAgent():
    def __init__(self, c, start):
        self.cur = c
        self.start  = start

    def _apply_hazard(self):
        # Teleport
        if self.cur.type == "teleport" and self.cur.tpdest is not None:
            print(f"Teleported from {self.cur.pos} -> {self.cur.tpdest.pos}")
            self.cur = self.cur.tpdest

        # Death pit
        elif self.cur.type == "deathpit":
            print(f"Fell into death pit at {self.cur.pos}")
            self.cur = start
        # Confusion handled in connect(), so nothing needed here

    def move_up(self):
        if self.cur.up is not None:
            self.cur = self.cur.up
            self._apply_hazard()

    def move_down(self):
        if self.cur.down is not None:
            self.cur = self.cur.down
            self._apply_hazard()

    def move_left(self):
        if self.cur.left is not None:
            self.cur = self.cur.left
            self._apply_hazard()

    def move_right(self):
        if self.cur.right is not None:
            self.cur = self.cur.right
            self._apply_hazard()

    def get_pos(self):
        return self.cur.pos

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
    #solves hazardless maze with bfs
    m = solve_maze("MAZE_0.png")
    
    # MAZE_1
    cells, hazards, start, goal_pos = ml.getHMaze("MAZE_1.png")
    #returns array with all the cells in the maze, 64x64, hazards list for interactions, start node for agent, and goal position
    #starting position for test

    # agent = testAgent(cells[1][17], start) #confusion test

    # agent = testAgent(cells[6][30], start) # teleport test, should end up at 59,55

    agent = testAgent(cells[9][5], start) # deathpit test, should end up at start (0,31)

    print("Start position:", agent.get_pos())

    agent.move_down()
    #agent.move_up() #confusion test only, should move down isntead and end up at 3,17

    print("End position:", agent.get_pos())