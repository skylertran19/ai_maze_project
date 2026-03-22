from PIL import Image, ImageDraw
import numpy as np
from collections import deque

# Matrix codes:
# 0 = Empty/navigable cell
# 1 = Wall
# 2 = Start 'S' (empty space on outer edge)
# 3 = Goal/Exit 'G' (empty space on outer edge)
# 4 = Death pits 'P' 🔥
# 5 = Teleport pad 'T' 🟢#41d281 ✳️#3cc676 🟡#ecb331 ✴️#ff8536 🟣#714eb5 🔯#814de5
# 6 = Confusion pad 'C' 😵‍💫 (not specified as 6 in the instructions) #ffc534 face #5e3327 lines

TP_G = "green"      # Green teleport
TP_O = "orange"     # Orange teleport
TP_P = "purple"     # Purple teleport

def get_color_category(r, g, b):
    if r > 200 and g > 200 and b > 200: #navigable
        return (0, None)
    if r < 50 and g < 50 and b < 50: #black | wall
        return (1, None)
    
    # Teleport pads 🟡✴️ - orange-yellow R=255, check FIRST
    if r == 236 and 160 < g < 200 and 30 < b < 60: #rgb(236 179 49)
        return (5, TP_O)
    if r == 255 and 115 < g < 160 and 36 < b < 75: #rgb(255 133 54)
        return (5, TP_O)
    
    # Both death and confusion are brown, use R value as primary separator:
    # Death pit 🔥: R >= 159 (159, 159, 163, 167, 171, 172)
    # Confusion 😵‍💫: R <= 158 (122-158)
    
    # Death pit - brown with R >= 159
    if r >= 159 and 95 < g < 135 and 40 < b < 75:
        return (4, "fire")
    
    # Confusion - brown with R <= 158
    if r <= 158 and 75 < g < 135 and 60 < b < 75:
        return (6, "confusion")
    
    # Green teleports
    if g > 180 and r < 100 and b > 80 and b < 160:
        return (5, TP_G)
    
    # Purple teleports
    if 100 < r < 140 and g < 100 and b > 150:
        return (5, TP_P)
    
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
    """Load hazards from image and return clustered coordinates."""
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
    
    death_pits, teleports, confusion_pits = load_hazards_from_image("MAZE_1.png")
    
    # Add hazard codes directly from original image pixels to maze
    img = Image.open("MAZE_1.png").convert("RGB")
    img_array = np.array(img)
    
    for r in range(img_array.shape[0]):
        for c in range(img_array.shape[1]):
            pixel = img_array[r, c]
            code, _ = get_color_category(pixel[0], pixel[1], pixel[2])
            
            if code in [4, 5, 6]:
                scaled_r, scaled_c = scale_to_64x64(r, c)
                maze[scaled_r, scaled_c] = code
    
    # Print hazard locations from the updated maze array
    print("\n=== Hazards in Maze ===")
    
    death_pit_coords = list(zip(*np.where(maze == 4)))
    print(f"\nDeath Pits ({len(death_pit_coords)}):")
    for coord in sorted(death_pit_coords):
        print(f"  {coord}")
    
    teleport_coords = list(zip(*np.where(maze == 5)))
    print(f"\nTeleport Pads ({len(teleport_coords)}):")
    for coord in sorted(teleport_coords):
        print(f"  {coord}")
    
    confusion_coords = list(zip(*np.where(maze == 6)))
    print(f"\nConfusion Pads ({len(confusion_coords)}):")
    for coord in sorted(confusion_coords):
        print(f"  {coord}")