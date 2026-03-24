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

def scale_to_64x64(r, c, img_size=1026):
    #assuming all maze images will be 1026 x 1026
    scaled_r = int(r * 64 / img_size)
    scaled_c = int(c * 64 / img_size)
    # Clamp to valid range
    scaled_r = min(scaled_r, 63)
    scaled_c = min(scaled_c, 63)
    return scaled_r, scaled_c

#Load maze image and converts to binary grid (1=wall,0=path).
def loadmaze(filename, threshold=128):
    img = Image.open(filename).convert("L")  # grayscale
    maze = (np.array(img) < threshold).astype(int)
    return maze, img