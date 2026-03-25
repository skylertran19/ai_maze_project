from PIL import Image, ImageDraw
import numpy as np
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

#NO HAZARDS Load maze image and converts to binary grid (1=wall,0=path).
def loadmaze(filename, threshold=128):
    img = Image.open(filename).convert("L")  # grayscale
    maze = (np.array(img) < threshold).astype(int)
    return maze, img

#color lookup dictionaries from distinct_colors_dict
COLOR_TO_HAZARD = {}
for hazard_name, colors in DISTINCT_COLORS.items():
    for color in colors:
        COLOR_TO_HAZARD[color] = hazard_name

def get_color_category(r, g, b):#returns (code, type)
    color = (r, g, b)
    # Check if color is in the distinct colors dictionary
    if color in COLOR_TO_HAZARD:
        hazard_name = COLOR_TO_HAZARD[color]
        
        if hazard_name == "deathpit":
            return (4, "deathpit")
        elif hazard_name == "confusion":
            return (6, "confusion")
        elif hazard_name == "greentp" or hazard_name == "greentpdest":
            return (5, TP_G)
        elif hazard_name == "yellowtp" or hazard_name == "orangetpdest":
            return (5, TP_O)
        elif hazard_name == "purpletp" or hazard_name == "purpletpdest":
            return (5, TP_P)
    
    if r > 250 and g > 250 and b > 250:  # navigable (light)
        return (0, None)
    if r < 10 and g < 10 and b < 10:  # black | wall
        return (1, None)
    
    return (0, None)

def cluster_nearby_pixels(pixels_list, max_distance=15):
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

def scale_to_64x64(r, c, img_size=1026):
    #assuming all maze images will be 1026 x 1026
    scaled_r = int(r * 64 / img_size)
    scaled_c = int(c * 64 / img_size)
    # Clamp to valid range
    scaled_r = min(scaled_r, 63)
    scaled_c = min(scaled_c, 63)
    return scaled_r, scaled_c

def printHazards(hazard_locations):
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

def detectHazards(img_array): #returns lists of hazards types with coordinates
    #returns the coordinates in 1026x1026
    death_pit_pixels = []
    confusion_pit_pixels = []
    teleport_pixels = []
    
    rows, cols = img_array.shape[0], img_array.shape[1]
    
    for r in range(rows):
        for c in range(cols):
            pixel = img_array[r, c]
            code, color_info = get_color_category(pixel[0], pixel[1], pixel[2])
            
            if code == 4:
                death_pit_pixels.append((r, c))
            elif code == 5:
                teleport_pixels.append((r, c))
            elif code == 6:
                confusion_pit_pixels.append((r, c))
    
    # Cluster pixels into individual hazards
    death_pits = cluster_nearby_pixels(death_pit_pixels, max_distance=14)
    confusion_pits = cluster_nearby_pixels(confusion_pit_pixels, max_distance=14)
    teleports = cluster_nearby_pixels(teleport_pixels, max_distance=14)
    
    return death_pits, teleports, confusion_pits

def loadHazardsMaze(filename):
    # Returns a 129x129 maze: 2px outer border, then 14px passage | 2px wall repeating.
    #   odd rows/cols  = passage cells (0 open, or hazard code 2-6)
    #   even rows/cols = wall/opening cells (1=wall, 0=opening), corner cells (even,even) always = 1
    N = 64
 
    def passage_px(i):  return 9 + i * 16
    def wall_px(i):     return 16 + i * 16
 
    img_gray  = Image.open(filename).convert("L")
    binary = (np.array(img_gray) < 128).astype(float)  # 1=wall pixel
    img_rgb = Image.open(filename).convert("RGB")
    img_array = np.array(img_rgb)
    mazehires = (np.array(img_gray) < 128).astype(int)
 
    thick = np.ones((2*N+1, 2*N+1), dtype=int)
 
    for r in range(N):           # passage cells always open
        for c in range(N):
            thick[2*r+1, 2*c+1] = 0
 
    for r in range(N-1):         # horizontal wall/opening between row r and r+1
        wp = wall_px(r)
        for c in range(N):
            thick[2*r+2, 2*c+1] = 1 if binary[wp, passage_px(c)] > 0.5 else 0
 
    for r in range(N):           # vertical wall/opening between col c and c+1
        rp = passage_px(r)
        for c in range(N-1):
            thick[2*r+1, 2*c+2] = 1 if binary[rp, wall_px(c)] > 0.5 else 0
 
    # Outer border walls
    thick[0, :] = 1; thick[-1, :] = 1
    thick[:, 0] = 1; thick[:, -1] = 1
 
    #Mark start(2) and Goal(3)
    for c in range(N):
        cp = passage_px(c)
        if binary[0,  cp] < 0.5: thick[0,  2*c+1] = 2; thick[1,    2*c+1] = 2
        if binary[-1, cp] < 0.5: thick[-1, 2*c+1] = 3; thick[-2,   2*c+1] = 3
    for r in range(N):
        rp = passage_px(r)
        if binary[rp,  0] < 0.5: thick[2*r+1, 0]  = 2; thick[2*r+1, 1]   = 2
        if binary[rp, -1] < 0.5: thick[2*r+1, -1] = 3; thick[2*r+1, -2]  = 3
 
    #Hazard overlay
    hazard_locations = {4: set(), 5: {}, 6: set()}
    rows, cols = img_array.shape[0], img_array.shape[1]
 
    for r in range(rows):
        for c in range(cols):
            pixel = img_array[r, c]
            code, hazard_type = get_color_category(pixel[0], pixel[1], pixel[2])
            if code in [4, 5, 6]:
                pr = min(max(round((r - 9) / 16), 0), N-1)  # passage row index (0-63)
                pc = min(max(round((c - 9) / 16), 0), N-1)  # passage col index (0-63)
                thick[2*pr+1, 2*pc+1] = code
                if code == 5:
                    hazard_locations[5][(pr, pc)] = hazard_type
                else:
                    hazard_locations[code].add((pr, pc))
 
    render_hazards(mazehires, hazard_locations, filename)
    printHazards(hazard_locations)
    return thick, hazard_locations
 
def render_hazards(maze, hazard_locations, filename):
    #Render hazards on top of a black/white maze visualization.
    # 1026x1026 numpy array 
    # hazard_locations: keys 4, 5, 6 containing hazard coordinates (64x64)
    teleport_color_map = {
        TP_G: (0, 166, 0),
        TP_O: (255, 133, 0),
        TP_P: (155, 9, 255),
    }
    rows, cols = maze.shape
 
    #Base
    img = Image.new("RGB", (cols, rows))
    pixels = img.load()
    
    for r in range(rows):
        for c in range(cols):
            code = maze[r, c]
            if code == 1:  # Wall
                color = (0, 0, 0)
            else:  # Space, hazards
                color = (255, 255, 255)
            pixels[c, r] = color
    
    #layer hazards on top
    draw = ImageDraw.Draw(img)
    radius = 12  # Adjust as needed for 1026x1026 scale
    # Death pits - red filled circles
    for (scaled_r, scaled_c) in hazard_locations[4]:
        # Convert 1026x1026 for visualization
        hires_r = int(scaled_r * 1026 / 64) + 3
        hires_c = int(scaled_c * 1026 / 64) + 3
        bbox = [hires_c , hires_r , hires_c + radius, hires_r + radius]
        draw.ellipse(bbox, fill=(255, 0, 0), outline=(255, 0, 0))
    
    # Confusion pits - dark yellow filled squares
    for (scaled_r, scaled_c) in hazard_locations[6]:
        hires_r = int(scaled_r * 1026 / 64) + 2.5
        hires_c = int(scaled_c * 1026 / 64) + 2.5
        bbox = [hires_c , hires_r , hires_c + radius, hires_r + radius]
        draw.rectangle(bbox, fill=(200, 200, 0), outline=(200, 200, 0))
    
    # Teleports - colored filled circles
    for (scaled_r, scaled_c), color_type in hazard_locations[5].items():
        hires_r = int(scaled_r * 1026 / 64) + 2.5
        hires_c = int(scaled_c * 1026 / 64) + 2.5
        
        if color_type in teleport_color_map:
            color = teleport_color_map[color_type]
        else:
            color = (100, 100, 100)
        
        bbox = [hires_c , hires_r , hires_c + radius, hires_r + radius]
        draw.ellipse(bbox, fill=color, outline=color)
    
    output_path = filename.replace(".png", "vis.png")
    img.save(output_path)
    return img

class cell:
    #single position in the 64x64
    __slots__ = ('pos', 'type', 'tpcolor', 'tpdest', 'right', 'left', 'up', 'down')
 
    def __init__(self, pos, cell_type, tpcolor=None, tpdest=None):
        self.pos     = pos        # (row, col) in 64x64 grid
        self.type    = cell_type  # "empty" | "start" | "goal" | "deathpit" | "teleport" | "confusion"
        self.tpcolor = tpcolor    # teleport colour string, or None
        self.tpdest  = tpdest     # destination cell object if teleport
        self.right = None #none means there is a wall in this direction
        self.left  = None
        self.up    = None
        self.down  = None

    #may implement this somewhere else
    def connect(self, direction, neighbour):
        if self.type == "confusion":
            flipped = {"up": "down", "down": "up", "left": "right", "right": "left"}
            direction = flipped[direction]
        setattr(self, direction, neighbour)
 
 
def getHMaze(filename):
    # node array 64x64 array of cells for agent to talk, haz dict, start cell, and goal position
    thick, haz = loadHazardsMaze(filename)
    N = 64
 
    # --- Pass 1: create all cell objects ---
    nodes    = np.empty((N, N), dtype=object)
    start    = None
    goal_pos = None
 
    for pr in range(N):
        for pc in range(N):
            tr, tc = 2*pr+1, 2*pc+1   # thick-maze
            code = thick[tr, tc]
 
            tpcolor = haz[5].get((pr, pc)) if code == 5 else None
 
            if   code == 0: cell_type = "empty"
            elif code == 2: cell_type = "start"
            elif code == 3: cell_type = "goal"
            elif code == 4: cell_type = "deathpit"
            elif code == 5: cell_type = "teleport"
            elif code == 6: cell_type = "confusion"
            else:           cell_type = "empty"
 
            n = cell((pr, pc), cell_type, tpcolor=tpcolor)
            nodes[pr, pc] = n
 
            if cell_type == "start":
                start = n
            elif cell_type == "goal":
                goal_pos = (pr, pc)
 
    # connecting neighbors 0 = open passage, 1 = wall (no connection)
    DIRS = {
        "up":    (-1,  0),
        "down":  ( 1,  0),
        "left":  ( 0, -1),
        "right": ( 0,  1),
    }
    for pr in range(N):
        for pc in range(N):
            n = nodes[pr, pc]
            for direction, (dr, dc) in DIRS.items():
                nr, nc = pr + dr, pc + dc
                if not (0 <= nr < N and 0 <= nc < N):
                    continue  # boundary — leave neighbour as None
                edge_tr = (2*pr+1) + dr
                edge_tc = (2*pc+1) + dc
                if thick[edge_tr, edge_tc] == 0:   # opening exists
                    n.connect(direction, nodes[nr, nc])
 
    #add teleport destinations; tpdest points to other pad of the same colour
    # Non-teleport cells leave tpdest as None.
    tp_by_color = {}
    for (pr, pc), color in haz[5].items():
        tp_by_color.setdefault(color, []).append((pr, pc))
 
    for color, coords in tp_by_color.items():
        if len(coords) == 2:
            src, dst = coords # src.tpdest points to teleport destination
            nodes[src].tpdest = nodes[dst]
            nodes[dst].tpdest = nodes[src]
    return nodes, haz, start, goal_pos