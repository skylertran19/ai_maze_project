from PIL import Image
import os
import csv
from pathlib import Path
from collections import defaultdict

def load_images_and_colors(image_folder):
    """Load all PNG images and extract their colors."""
    image_colors = {}
    
    png_files = sorted(Path(image_folder).glob("*.png"))
    
    for image_path in png_files:
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            colors = set(img.getdata())
            image_colors[image_path.name] = colors
            print(f"✓ {image_path.name:20} ({img.size[0]:2}x{img.size[1]:2}) - {len(colors):3} colors")
        except Exception as e:
            print(f"✗ Error loading {image_path.name}: {e}")
    
    return image_colors

def find_distinct_colors(image_colors):
    """Find colors unique to each image."""
    distinct_colors = defaultdict(set)
    
    for image_name, colors in image_colors.items():
        other_colors = set()
        for other_name, other_color_set in image_colors.items():
            if other_name != image_name:
                other_colors.update(other_color_set)
        
        distinct_colors[image_name] = colors - other_colors
    
    return distinct_colors


def export_to_python(image_colors, distinct_colors, filename="distinct_colors_dict.py"):
    """Export to Python dictionary format."""
    with open(filename, 'w') as f:
        f.write("# Distinct colors for each hazard symbol\n")
        f.write("DISTINCT_COLORS = {\n")
        
        for image_name in sorted(image_colors.keys()):
            clean_name = image_name.replace('.png', '')
            colors_list = sorted(list(distinct_colors[image_name]))
            f.write(f"    '{clean_name}': [\n")
            for rgb in colors_list:
                f.write(f"        {rgb},\n")
            f.write("    ],\n")
        
        f.write("}\n")
    

def main():
    """Main program."""
    image_folder = "hazards symbols"
    
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' not found!")
        return
    
    
    image_colors = load_images_and_colors(image_folder)
    
    if not image_colors:
        return
    
    distinct_colors = find_distinct_colors(image_colors)
    
    export_to_python(image_colors, distinct_colors)
    
if __name__ == "__main__":
    main()
