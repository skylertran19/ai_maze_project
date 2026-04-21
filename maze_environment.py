#transforms maze matrix and hazard information into maze environment
import numpy as np

class cell:
    #single position in the 64x64
    __slots__ = ('pos', 'type', 'tpcolor', 'tpdest', 'right', 'left', 'up', 'down')
 
    def __init__(self, pos, cell_type, tpcolor=None, tpdest=None):
        self.pos     = pos        # (row, col) in 64x64 grid
        self.type    = cell_type  # "empty" | "start" | "goal" | "deathpit" | "teleport" | "confusion"
        self.tpcolor = tpcolor    # teleport color string, or None
        self.tpdest  = tpdest     # destination cell object if teleport
        self.right = None #none means there is a wall in this direction
        self.left  = None
        self.up    = None
        self.down  = None

    def connect(self, direction, neighbour):
        if self.type == "confusion":
            flipped = {"up": "down", "down": "up", "left": "right", "right": "left"}
            direction = flipped[direction]
        setattr(self, direction, neighbour)

#assume its been given: array with all the cells in the maze, 64x64, hazards list 
#receive an image, of walls only, or all hazards minus deathpits
#then render 4 images of the map with hazards for each rotation of the death pits
#this class will figure out the vertex  of the deathpit clump
#assumme that no deathpit clumps will overlap, so the vertex will always be the only spot with other deathpics
#in more than one diagonal direction, and will always be on the map, even if one of its arms extends out of the map
# 0 = Empty/navigable cell
# 1 = Wall
# 2 = Start 'S' (empty space on outer edge)
# 3 = Goal/Exit 'G' (empty space on outer edge)
# 4 = Death pits 'P' 🔥
# 5 = Teleport pad 'T' 🟢✳️ 🟡✴️ 🟣🔯#
# 6 = Confusion pad 'C' 😵‍💫
class Maze:
    def __init__(self, cells, hazards, start, goal_pos, base_image):
        self.cells = cells
        self.hazards = hazards
        self.start = start
        self.goal_pos = goal_pos
        self.cur = start
        self.base_image = base_image
        self.cur_rotation = 0

        #figure out death printHazards
    
        #render 4 images of wherever the current hazards are

    def _apply_deathpits(self):
        self.death_vertices = []
        # deathpits take over an area of 7x4

        for coord in self.hazards[4]:
            r, c = coord
            diag_count = 0
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.hazards[4]:
                    diag_count += 1
            if diag_count > 1:
                self.death_vertices.append(coord)

        # need to account when the vertex is on the edge of the map
        #only showing one arm atm, only one vertex in a large area, etc
    
    def deathpit_direction(self, vertex):
        if(vertex not in self.hazards[4]):
            return -1
        r, c = vertex

        #regular vertices; they have deathpits in 2 diagonal directions
        if (r - 1, c + 1) in self.hazards[4]:
            if(r + 1, c + 1) in self.hazards[4]: #v
                return 0
            elif (r - 1, c - 1) in self.hazards[4]: #>
                return 3
            #later
        if (r + 1, c + 1) in self.hazards[4]:
            if(r + 1, c - 1) in self.hazards[4]: #^
                return 2
            elif (r - 1, c + 1) in self.hazards[4]: #<
                return 1
            
        #need to account for deathpits only in one diagnonal direction


        #only one deathpit, assuming there will be no deathpits at 0,0, 0,63, 63,0, or 63,63
        if 0 < r < 63 and c == 0: return 0 #v
        if 0 < r < 63 and c == 63: return 1 # <
        if r == 63 and 0 < c < 63: return 2 #^
        if r == 0 and 0 < c < 63: return 3 #>

        return -1
        
        
        


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

