from PIL import Image
import numpy as np
def maze_loader(filename, threshold=128):
    img = Image.open(filename).convert("L")
    maze_array = np.array(img)
    maze = (maze_array < threshold).astype(int)
    return maze

maze = maze_loader("MAZE_0.png")
print(maze)