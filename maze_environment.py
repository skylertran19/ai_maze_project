#transforms maze matrix and hazard information into maze environment
from typing import List, Tuple
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw

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

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

class TurnResult:
    wall_hits: int # +1 for each action that attempts invalid movement, Number of wall collisions this turn (0-5)
    #Hitting a wall does NOT change agent position.
    current_position: Tuple[int, int] # Agent's (x, y) coordinates after all actions executed
    # If dead: position of death pit, If teleported: destination coordinates.
    is_dead: bool # True if agent stepped on death pit during this turn, Agent immediately respawns at start position next turn.
    is_goal_reached: bool # True if agent reached exit
    #Episode ends immediately. Triggers success metrics calculation.
    teleported: bool # True if teleport was triggered this turn, current_position shows post-teleport location.
    actions_executed: int # Number of actions completed beforedeath/goal
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0

class MazeEnvironment:
    #start_pos : Tuple[int, int]
    def __init__(self, maze_id: str, cells, hazards, start_pos, goal_pos, base_image):
        self.maze_id : str = maze_id #'training' or 'testing'
        self.cells = cells
        self.hazards = hazards
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.cur = start_pos
        self.base_image_filename = base_image
        self.cur_rotation = 0

        self._apply_deathpits() #self.death_vertices
    
    
    def reset(self) -> Tuple[int, int]:
        #Reset environment for new episode
        #Returns: Starting position coordinates
        self.cur_rotation = 0
        self.cur = self.start_pos
        return self.cur.pos
    
    def step(self, actions: List[Action]) -> TurnResult:
        """
        Execute a turn with given actions
        Args:
        actions: List of 1-5 Action objects
        Returns:
        TurnResult with feedback
        Raises:
        ValueError: If actions list empty or >5 actions
        """
        if not actions or len(actions) > 5:
            raise ValueError("Actions list must contain 1-5 actions")
        
        result = TurnResult()
        
        for action in actions:
            result.actions_executed += 1
            
            # Map action to direction and attempt movement
            direction_map = {
                Action.MOVE_UP: 'up',
                Action.MOVE_DOWN: 'down',
                Action.MOVE_LEFT: 'left',
                Action.MOVE_RIGHT: 'right',
            }
            if action == Action.WAIT:
                #wait
                pass
            elif action in direction_map:
                direction = direction_map[action]
                neighbor = getattr(self.cur, direction)
                if neighbor is None: #if wall
                    result.wall_hits += 1
                else:
                    self.cur = neighbor
            
            # Apply hazard effects after movement
            if self.cur.type == "deathpit":
                result.is_dead = True
                self.cur = self.start_pos  # respawn at start cell
                break  # end episode
            
            elif self.cur.type == "teleport" and self.cur.tpdest is not None:
                self.cur = self.cur.tpdest
                result.teleported = True
            
            elif self.cur.type == "confusion":
                result.is_confused = True
                # direction flip due to confusion already implemented
                # into the cell, right/left/up/down pointers from maze construction
            
            elif self.cur.type == "goal":
                result.is_goal_reached = True
                break
        
        result.current_position = self.cur.pos
        
        #NEED TO ADD DEATHPIT ROTATIONS STILL
        # self._rotate_deathpits()
        
        return result
    def get_episode_stats(self) -> dict:
        """
        Get statistics for current episode
        """
        #lowk what kind of statistics
        pass
    
    def _rotate_deathpits(self):
        #rotate clockwise, change cells
        pass
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

        #need to account when the vertex is on the edge of the map
        #or only showing one arm atm, only one vertex in a large area, etc

    def _render_deathpits(self):
        img = Image.open(self.base_image_filename).copy()
        draw = ImageDraw.Draw(img)
        radius = 12  #same as render_hazards in maze_loader
        
        # Draw red circles for each deathpit
        for (scaled_r, scaled_c) in self.hazards[4]:
            # Convert from 64x64 to 1026x1026 coordinates
            hires_r = int(scaled_r * 1026 / 64) + 3
            hires_c = int(scaled_c * 1026 / 64) + 3
            
            # Create bounding box for the circle
            bbox = [hires_c, hires_r, hires_c + radius, hires_r + radius]
            draw.ellipse(bbox, fill=(255, 0, 0), outline=(255, 0, 0))
        
        output_path = self.base_image_filename.replace(".png", f"_rotation{self.cur_rotation}.png")
        img.save(output_path)
        
        return img
    
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
        #ignoring rn since maze1 has complete deathpits

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
# 5 = Teleport pad 'T' 🟢✳️ 🟡✴️ 🟣🔯
# 6 = Confusion pad 'C' 😵‍💫
