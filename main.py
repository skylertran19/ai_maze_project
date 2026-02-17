from PIL import Image
import numpy as np

def load_maze_from_png(filename, threshold=128):
    """
    Load a maze from a PNG image.
    
    Args:
        filename: Path to the PNG file.
        threshold: Pixel value threshold to distinguish wall vs path.
                   Pixels darker than threshold -> wall (1)
                   Pixels lighter than threshold -> path (0)
                   
    Returns:
        maze: 2D NumPy array (1 = wall, 0 = path)
    """
    # Load image and convert to grayscale
    img = Image.open(filename).convert("L")  # "L" = grayscale
    
    # Convert to NumPy array
    maze_array = np.array(img)
    
    # Binarize: wall = 1, path = 0
    maze = (maze_array < threshold).astype(int)
    
    return maze

# Example usage
maze = load_maze_from_png("MAZE_0.png")
print(maze)

