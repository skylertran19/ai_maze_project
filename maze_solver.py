from PIL import Image, ImageDraw
import numpy as np
from collections import deque
from distinct_colors_dict import DISTINCT_COLORS

# Matrix codes:
# 0 = Empty/navigable cell
# 1 = Wall
# 2 = Start 'S' (empty space on outer edge)
# 3 = Goal/Exit 'G' (empty space on outer edge)
# 4 = Death pits 'P' 🔥
# 5 = Teleport pad 'T' 🟢#41d281 ✳️#3cc676 🟡#ecb331 ✴️#ff8536 🟣#714eb5 🔯#814de5
# 6 = Confusion pad 'C' 😵‍💫 (not specified as 6 in the instructions) #ffc534 face #5e3327 lines
TP_G = "green"      # 🟢 → ✳️
TP_O = "orange"    # 🟡 → ✴️
TP_P = "purple"    # 🟣 → 🔯

# Build color lookup dictionaries from distinct_colors_dict
COLOR_TO_HAZARD = {}
for hazard_name, colors in DISTINCT_COLORS.items():
    for color in colors:
        COLOR_TO_HAZARD[color] = hazard_name

def get_color_category(r, g, b):
    color = (r, g, b)
    # Check if color is in the distinct colors dictionary
    if color in COLOR_TO_HAZARD:
        hazard_name = COLOR_TO_HAZARD[color]
        
        if hazard_name == "deathpit":
            return (4, "deathpit")
        elif hazard_name == "confusion":
            return (6, "confusion")
        elif hazard_name == "greentp" or hazard_name == "greentpdest":
            return (5, "TP_G")
        elif hazard_name == "yellowtp" or hazard_name == "orangetpdest":
            return (5, "TP_O")
        elif hazard_name == "purpletp" or hazard_name == "purpletpdest":
            return (5, "TP_P")
    
    if r > 250 and g > 250 and b > 250:  # navigable (light)
        return (0, None)
    if r < 10 and g < 10 and b < 10:  # black | wall
        return (1, None)
    
    return (0, None)

def cluster_nearby_pixels(pixels_list, max_distance=10):
    if not pixels_list:
        return []
    
    pixels_list = list(pixels_list)
    clusters = []
    used = set()
    
    for i, (r1, c1) in enumerate(pixels_list):
        if i in used:
            continue
        
        cluster = [(r1, c1)]
        used.add(i)
        
        for j, (r2, c2) in enumerate(pixels_list):
            if j not in used:
                distance = ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5
                if distance <= max_distance:
                    cluster.append((r2, c2))
                    used.add(j)
        
        avg_r = int(sum(r for r, c in cluster) / len(cluster))
        avg_c = int(sum(c for r, c in cluster) / len(cluster))
        clusters.append((avg_r, avg_c))
    
    return clusters

def load_hazards_from_image(filename):
    img = Image.open(filename).convert("RGB")
    img_array = np.array(img)
    
    death_pit_pixels = []
    confusion_pit_pixels = []
    teleport_pixels = []
    
    # Track what colors we actually detect
    detected_codes = {0: 0, 1: 0, 4: 0, 5: 0, 6: 0}
    sample_pixels = {4: [], 5: [], 6: []}
    
    rows, cols = img_array.shape[0], img_array.shape[1]
    
    for r in range(rows):
        for c in range(cols):
            pixel = img_array[r, c]
            code, color_info = get_color_category(pixel[0], pixel[1], pixel[2])
            detected_codes[code] = detected_codes.get(code, 0) + 1
            
            if code == 4:
                death_pit_pixels.append((r, c))
                if len(sample_pixels[4]) < 3:
                    sample_pixels[4].append((pixel[0], pixel[1], pixel[2]))
            elif code == 5:
                teleport_pixels.append((r, c))
                if len(sample_pixels[5]) < 3:
                    sample_pixels[5].append((pixel[0], pixel[1], pixel[2]))
            elif code == 6:
                confusion_pit_pixels.append((r, c))
                if len(sample_pixels[6]) < 3:
                    sample_pixels[6].append((pixel[0], pixel[1], pixel[2]))
    
    # Cluster pixels into individual hazards
    death_pits = cluster_nearby_pixels(death_pit_pixels, max_distance=10)
    confusion_pits = cluster_nearby_pixels(confusion_pit_pixels, max_distance=10)
    teleports = cluster_nearby_pixels(teleport_pixels, max_distance=10)
    
    return death_pits, teleports, confusion_pits

def scale_to_64x64(r, c, img_size=1026):
    """Scale coordinates from full image to 64x64 grid."""
    scaled_r = int(r * 64 / img_size)
    scaled_c = int(c * 64 / img_size)
    # Clamp to valid range
    scaled_r = min(scaled_r, 63)
    scaled_c = min(scaled_c, 63)
    return scaled_r, scaled_c

#Load maze image and converts to binary grid (1=wall,0=path).
def maze_loader(filename, threshold=128):
    img = Image.open(filename).convert("L")  # grayscale
    maze = (np.array(img) < threshold).astype(int)
    return maze, img
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
    maze, original_img = maze_loader(filename)
    rows, cols = maze.shape
    #find entrance and exit
    start_col = np.where(maze[0] == 0)[0][len(np.where(maze[0] == 0)[0]) // 2]
    end_col = np.where(maze[-1] == 0)[0][len(np.where(maze[-1] == 0)[0]) // 2]
    start = (0, int(start_col))
    end = (rows - 1, int(end_col))
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
    
    # Load hazards from MAZE_1
    print("\n=== Loading Hazards from MAZE_1 ===")
    death_pits, teleports, confusion_pits = load_hazards_from_image("MAZE_1.png")
    
    img = Image.open("MAZE_1.png").convert("RGB")
    img_array = np.array(img)
    
    # Dictionary to track hazard locations in 64x64 grid
    hazard_locations = {
        4: set(),  # death pits
        5: set(),  # teleports
        6: set()   # confusion
    }
    
    rows, cols = img_array.shape[0], img_array.shape[1]
    
    
    for r in range(rows):
        for c in range(cols):
            pixel = img_array[r, c]
            code, hazard_type = get_color_category(pixel[0], pixel[1], pixel[2])
            
            if code in [4, 5, 6]:
                scaled_r, scaled_c = scale_to_64x64(r, c, img_size=cols)
                maze[scaled_r, scaled_c] = code
                hazard_locations[code].add((scaled_r, scaled_c))
    
    # Print hazard locations from the updated maze array
    print("\n=== Hazards in Maze (0-63 coordinates) ===")
    
    print(f"\nDeath Pits ({len(hazard_locations[4])}):")
    if hazard_locations[4]:
        for coord in sorted(hazard_locations[4]):
            print(f"  {coord}")
    else:
        print("  None")
    
    print(f"\nTeleport Pads ({len(hazard_locations[5])}):")
    if hazard_locations[5]:
        for coord in sorted(hazard_locations[5]):
            print(f"  {coord}")
    else:
        print("  None")
    
    print(f"\nConfusion Pads ({len(hazard_locations[6])}):")
    if hazard_locations[6]:
        for coord in sorted(hazard_locations[6]):
            print(f"  {coord}")
    else:
        print("  None")