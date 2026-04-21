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
        self.is_confused = False
        self.confused_turns_left = 0
        self._apply_deathpits() #finds vertices coords and their directions: self.death_vertices
        self._compute_rotations() #calculate locations of future deathpits, saves them in self.rotation_sets = []
    
    def reset(self) -> Tuple[int, int]:
        #Reset environment for new episode
        #Returns: Starting position coordinates
        old_set = self.rotation_sets[self.cur_rotation]
        new_set = self.rotation_sets[0]
        for r, c in old_set - new_set:
            self.cells[r][c].type = "empty"
        for r, c in new_set - old_set:
            self.cells[r][c].type = "deathpit"
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
 
        FLIP = {"up": "down", "down": "up", "left": "right", "right": "left"}
        direction_map = {
            Action.MOVE_UP: 'up',
            Action.MOVE_DOWN: 'down',
            Action.MOVE_LEFT: 'left',
            Action.MOVE_RIGHT: 'right',
        }

        if self.is_confused:
            result.is_confused = True
        
        for action in actions:
            result.actions_executed += 1
 
            if action == Action.WAIT:
                pass
            elif action in direction_map:
                direction = direction_map[action]
                if self.is_confused:
                    direction = FLIP[direction]  # flip intended direction
                neighbor = getattr(self.cur, direction)
                if neighbor is None:
                    result.wall_hits += 1
                else:
                    self.cur = neighbor
 
            # Apply hazard effects after movement
            if self.cur.type == "deathpit":
                result.is_dead = True
                self.cur = self.start_pos
                break
 
            elif self.cur.type == "teleport" and self.cur.tpdest is not None:
                self.cur = self.cur.tpdest
                result.teleported = True
 
            elif self.cur.type == "confusion":
                # Confused for remainder of this turn + all of next turn
                self.is_confused = True
                self.confused_turns_remaining = 2
                result.is_confused = True
 
            elif self.cur.type == "goal":
                result.is_goal_reached = True
                break
 
        # Tick confusion counter down at end of turn
        if self.is_confused:
            self.confused_turns_remaining -= 1
            if self.confused_turns_remaining <= 0:
                self.is_confused = False
 
        result.current_position = self.cur.pos
        self._rotate_deathpits()
        
        return result
    
    def get_episode_stats(self) -> dict:
        """
        Get statistics for current episode
        """
        #lowk what kind of statistics
        pass

    def _apply_deathpits(self): #find vertices of deathpit clumps
        self.death_vertices = []
        # deathpits take over an area of 7x4

        for coord in self.hazards[4]:
            r, c = coord
            diag_neighbors = [(r+dr, c+dc) for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]
                              if (r+dr, c+dc) in self.hazards[4]]
            for i in range(len(diag_neighbors)):
                for j in range(i+1, len(diag_neighbors)):
                    dr1, dc1 = diag_neighbors[i][0]-r, diag_neighbors[i][1]-c
                    dr2, dc2 = diag_neighbors[j][0]-r, diag_neighbors[j][1]-c
                    if (dr1 == dr2 and dc1 != dc2) or (dc1 == dc2 and dr1 != dr2):
                        self.death_vertices.append((coord, self.deathpit_direction(coord)))#vertex coordinates AND its direction
                        break 
        #need to account when the vertex is on the edge of the map
        #or only showing one arm atm, only one vertex in a large area, etc
    
    def _compute_rotations(self): #do in constructor, to easily switch each turn
        #Saves the coordinates of all deathpits, stored in rotation_sets[x], x being the rotation it belongs to
        #rotation 0
        self.rotation_sets = {0: set(self.hazards[4])}

        ARM_DIRS = {
            "up":    [(-1,-1), (-1,+1)],
            "right": [(-1,+1), (+1,+1)],
            "down":  [(+1,-1), (+1,+1)],
            "left":  [(-1,-1), (+1,-1)],
        }
        CLOCKWISE = ["up", "right", "down", "left"]
        ARM_LENGTH = 3

        #rotation 1 - 3
        for rotation in range(1, 4):
            coords = set()
            for vertex, base_dir in self.death_vertices:
                vr, vc = vertex
                coords.add(vertex)
                rot_dir = CLOCKWISE[(CLOCKWISE.index(base_dir) + rotation) % 4]
                for dr, dc in ARM_DIRS[rot_dir]:
                    for step in range(1, ARM_LENGTH + 1):
                        nr, nc = vr + dr*step, vc + dc*step
                        if 0 <= nr < 64 and 0 <= nc < 64:
                            coords.add((nr, nc))
            self.rotation_sets[rotation] = coords
            #get directions of current vertices
            #change directions to next rotation
            #calculate locations of their arms, save in rotation_sets

        #render images for the 4 rotations with the coordinates
        for rot in range(4):
            self._render_deathpits(rot)
 
    def _render_deathpits(self, rotation): #called by _compute_rotations, only do once
        # Draws deathpit circles for the given rotation onto the base image.
        # Uses self.rotation_sets[rotation] coordinates only, doesnt touch cells
        img = Image.open(self.base_image_filename).copy()
        draw = ImageDraw.Draw(img)
        radius = 12
 
        for (scaled_r, scaled_c) in self.rotation_sets[rotation]:
            hires_r = int(scaled_r * 1026 / 64) + 3
            hires_c = int(scaled_c * 1026 / 64) + 3
            bbox = [hires_c, hires_r, hires_c + radius, hires_r + radius]
            draw.ellipse(bbox, fill=(255, 0, 0), outline=(255, 0, 0))
 
        output_path = self.base_image_filename.replace("base.png", f"_rot{rotation}.png")
        img.save(output_path)
        return img
    
    def deathpit_direction(self, vertex):
        if vertex not in self.hazards[4]:
            return None
        r, c = vertex
        diag_neighbors = [(r+dr, c+dc) for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]
                          if (r+dr, c+dc) in self.hazards[4]]
        for i in range(len(diag_neighbors)):
            for j in range(i+1, len(diag_neighbors)):
                dr1, dc1 = diag_neighbors[i][0]-r, diag_neighbors[i][1]-c
                dr2, dc2 = diag_neighbors[j][0]-r, diag_neighbors[j][1]-c
                if dr1 == dr2 and dc1 != dc2:
                    return "down" if dr1 > 0 else "up"   # arms same row side
                if dc1 == dc2 and dr1 != dr2:
                    return "right" if dc1 > 0 else "left" # arms same col side
        return None
        
        #need to account for deathpits only in one diagnonal direction
        #ignoring rn since maze1 has clear deathpit vertices
    
    
    def _rotate_deathpits(self):
        #rotate clockwise from whatever we're on
        #CHANGE CELLS:
            #get rid of old deathpits(should only be the "left" arm?), and add new deathpits
            #dont need to touch the vertices but may do so for now for simplicity
        next_rot = (self.cur_rotation + 1) % 4
        old_set = self.rotation_sets[self.cur_rotation]
        new_set = self.rotation_sets[next_rot]
        for r, c in old_set - new_set:          # cells leaving deathpit zone
            self.cells[r][c].type = "empty"
        for r, c in new_set - old_set:          # cells entering deathpit zone
            self.cells[r][c].type = "deathpit"
        self.cur_rotation = next_rot

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
