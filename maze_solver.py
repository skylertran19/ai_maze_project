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
    death_pits, teleports, confusion_pits = ml.load_hazards_from_image("MAZE_1.png")
    
    img = Image.open("MAZE_1.png").convert("RGB")
    img_array = np.array(img)
    
    # Dictionary to track hazard locations in 64x64 grid
    hazard_locations = {
        4: set(),  # death pits
        5: {},     # teleports with color info
        6: set()   # confusion
    }
    
    rows, cols = img_array.shape[0], img_array.shape[1]
    
    for r in range(rows):
        for c in range(cols):
            pixel = img_array[r, c]
            code, hazard_type = ml.get_color_category(pixel[0], pixel[1], pixel[2])
            
            if code in [4, 5, 6]:
                scaled_r, scaled_c = ml.scale_to_64x64(r, c, img_size=cols)
                maze[scaled_r, scaled_c] = code
                
                if code == 5:  # teleport
                    hazard_locations[code][(scaled_r, scaled_c)] = hazard_type
                else:
                    hazard_locations[code].add((scaled_r, scaled_c))
    
    # Print hazard locations from the updated maze array
    print("\nHazard Coordinates")
    
    print(f"\nDeath Pits ({len(hazard_locations[4])}):")
    if hazard_locations[4]:
        for coord in sorted(hazard_locations[4]):
            print(f"  {coord}")
    else:
        print("  None")
    
    print(f"\nTeleport Pads ({len(hazard_locations[5])}):")
    if hazard_locations[5]:
        for coord in sorted(hazard_locations[5].keys()):
            color = hazard_locations[5][coord]
            print(f"  {coord}" + color)
    else:
        print("  None")
    
    print(f"\nConfusion Pads ({len(hazard_locations[6])}):")
    if hazard_locations[6]:
        for coord in sorted(hazard_locations[6]):
            print(f"  {coord}")
    else:
        print("  None")