# transforms maze matrix and hazard information into maze environment
import numpy as np
from typing import List, Tuple

class cell:
    # single position in the 64x64
    __slots__ = ('pos', 'type', 'tpcolor', 'tpdest', 'right', 'left', 'up', 'down')

    def __init__(self, pos, cell_type, tpcolor=None, tpdest=None):
        self.pos     = pos        # (row, col) in 64x64 grid
        self.type    = cell_type  # "empty" | "start" | "goal" | "deathpit" | "teleport" | "confusion"
        self.tpcolor = tpcolor    # teleport color string, or None
        self.tpdest  = tpdest     # destination cell object if teleport
        self.right = None         # None means there is a wall in this direction
        self.left  = None
        self.up    = None
        self.down  = None

    def connect(self, direction, neighbour):
        if self.type == "confusion":
            flipped = {"up": "down", "down": "up", "left": "right", "right": "left"}
            direction = flipped[direction]
        setattr(self, direction, neighbour)


# ---------------------------------------------
# Death-pit rotation helpers
# ---------------------------------------------

# Directions encoded as integers (matches existing deathpit_direction return values):
#   0 = V  (opens downward)
#   1 = <  (opens leftward)
#   2 = ^  (opens upward)
#   3 = >  (opens rightward)
# Rotating 90 deg CW cycles:  0 -> 3 -> 2 -> 1 -> 0  (V -> > -> ^ -> < -> V)
_CW_ROTATION = {0: 3, 3: 2, 2: 1, 1: 0}

def _arm_cells(vertex: Tuple[int, int], direction: int) -> List[Tuple[int, int]]:
    """
    Return all cells (including vertex) that belong to a V-shaped death-pit
    cluster given its vertex position and opening direction.

    The V shape is 7 rows x up to 4 cols (or transposed for horizontal).
    Layout for direction 0 (V opens downward), vertex at (r, c):

        row r:          (r, c)                          <- vertex (tip)
        row r+1:   (r+1,c-1)  (r+1,c+1)
        row r+2:  (r+2,c-2)            (r+2,c+2)
        row r+3:  (r+3,c-3)            (r+3,c+3)

    The other directions are 90 deg/180 deg/270 deg rotations of this layout.
    Cells that fall outside the 64x64 grid are silently omitted.
    """
    r, c = vertex

    # Relative offsets for the "V opens downward" canonical form
    # (dr, dc) pairs; vertex itself is (0,0)
    canonical = [
        (0,  0),
        (1, -1), (1,  1),
        (2, -2), (2,  2),
        (3, -3), (3,  3),
    ]

    # Rotation matrices for CW turns:
    #   0 rotations (V):  (dr, dc) -> ( dr,  dc)
    #   1 rotation  (<):  (dr, dc) -> ( dc, -dr)
    #   2 rotations (^):  (dr, dc) -> (-dr, -dc)
    #   3 rotations (>):  (dr, dc) -> (-dc,  dr)
    def rotate(dr, dc, turns):
        for _ in range(turns):
            dr, dc = dc, -dr
        return dr, dc

    # direction 0 = V (0 CW turns from canonical)
    # direction 3 = > (1 CW turn)
    # direction 2 = ^ (2 CW turns)
    # direction 1 = < (3 CW turns)
    turns_map = {0: 0, 3: 1, 2: 2, 1: 3}
    turns = turns_map[direction]

    cells = []
    for dr, dc in canonical:
        rdr, rdc = rotate(dr, dc, turns)
        nr, nc = r + rdr, c + rdc
        if 0 <= nr < 64 and 0 <= nc < 64:
            cells.append((nr, nc))
    return cells


def _find_vertices(pit_set: set) -> List[Tuple[int, int]]:
    """Return all vertex cells in the pit set (cells with >1 diagonal pit neighbour)."""
    vertices = []
    for (r, c) in pit_set:
        diag_count = sum(
            1 for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]
            if (r+dr, c+dc) in pit_set
        )
        if diag_count > 1:
            vertices.append((r, c))
    return vertices


def _deathpit_direction(vertex: Tuple[int, int], pit_set: set) -> int:
    """Determine the opening direction of a V-cluster given its vertex."""
    if vertex not in pit_set:
        return -1
    r, c = vertex

    if (r-1, c+1) in pit_set:
        if (r+1, c+1) in pit_set:
            return 0   # V
        if (r-1, c-1) in pit_set:
            return 3   # >
    if (r+1, c+1) in pit_set:
        if (r+1, c-1) in pit_set:
            return 2   # ^
        if (r-1, c+1) in pit_set:
            return 1   # <

    # Edge-of-map single-arm fallbacks
    if 0 < r < 63 and c == 0:  return 0
    if 0 < r < 63 and c == 63: return 1
    if r == 63 and 0 < c < 63: return 2
    if r == 0  and 0 < c < 63: return 3

    return -1


# ---------------------------------------------
# Maze class (used internally by MazeEnvironment)
# ---------------------------------------------

class Maze:
    """
    Holds the full cell graph and hazard metadata.
    Also owns the death-pit rotation state.
    """

    def __init__(self, cells, hazards, start, goal_pos):
        self.cells    = cells     # np.ndarray (64,64) of cell objects
        self.hazards  = hazards   # {4: set of (r,c), 5: {(r,c): color}, 6: set of (r,c)}
        self.start    = start     # cell object at start position
        self.goal_pos = goal_pos  # (r, c) tuple
        self.cur_rotation = 0     # 0-3 (increments each rotation event)

        # Decompose pit set into (vertex, direction, member_cells) triples
        self._pit_clusters: List[dict] = []
        self._init_pit_clusters()

    # -- Pit cluster initialisation ------------------------------------------

    def _init_pit_clusters(self):
        """Identify every V-cluster and record its initial state."""
        pit_set = set(self.hazards[4])
        visited = set()

        for v in _find_vertices(pit_set):
            direction = _deathpit_direction(v, pit_set)
            if direction == -1:
                continue
            members = _arm_cells(v, direction)
            self._pit_clusters.append({
                "vertex":    v,
                "direction": direction,
                "members":   set(members),
            })
            visited.update(members)

        # Any isolated pits not captured by a cluster
        for pos in pit_set - visited:
            self._pit_clusters.append({
                "vertex":    pos,
                "direction": -1,   # can't determine; won't rotate
                "members":   {pos},
            })

    # -- Rotation ------------------------------------------------------------

    def rotate_deathpits(self):
        """
        Rotate every V-cluster 90 deg clockwise around its vertex.
        Updates both the cell objects and hazards[4].
        Called by MazeEnvironment every 5 executed actions.
        """
        new_pit_set = set()

        for cluster in self._pit_clusters:
            if cluster["direction"] == -1:
                # Isolated pit - does not rotate
                new_pit_set.update(cluster["members"])
                continue

            vertex    = cluster["vertex"]
            direction = cluster["direction"]

            # Remove old deathpit type from cells
            for (r, c) in cluster["members"]:
                if self.cells[r, c].type == "deathpit":
                    self.cells[r, c].type = "empty"

            # Advance direction one CW step
            new_direction = _CW_ROTATION[direction]
            new_members   = set(_arm_cells(vertex, new_direction))

            # Apply new deathpit type to cells
            for (r, c) in new_members:
                if self.cells[r, c].type not in ("wall", "start", "goal"):
                    self.cells[r, c].type = "deathpit"

            cluster["direction"] = new_direction
            cluster["members"]   = new_members
            new_pit_set.update(new_members)

        self.hazards[4] = new_pit_set
        self.cur_rotation = (self.cur_rotation + 1) % 4


# ---------------------------------------------
# TurnResult
# ---------------------------------------------

class TurnResult:
    def __init__(self):
        self.wall_hits:       int              = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead:         bool             = False
        self.is_confused:     bool             = False
        self.is_goal_reached: bool             = False
        self.teleported:      bool             = False
        self.actions_executed: int             = 0


# ---------------------------------------------
# MazeEnvironment  <- main class students interact with
# ---------------------------------------------

from enum import Enum

class Action(Enum):
    MOVE_UP    = 0
    MOVE_DOWN  = 1
    MOVE_LEFT  = 2
    MOVE_RIGHT = 3
    WAIT       = 4

# Direction -> attribute name on a cell object
_DIR_ATTR = {
    Action.MOVE_UP:    "up",
    Action.MOVE_DOWN:  "down",
    Action.MOVE_LEFT:  "left",
    Action.MOVE_RIGHT: "right",
}

# Confusion inverts directions
_CONFUSED_MAP = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}


class MazeEnvironment:
    """
    Fully implemented maze environment.

    Hazard mechanics:
    -----------------
    - Death pits   - instant death, respawn at start next turn; V-cluster
                     rotates 90 deg CW every 5 executed actions.
    - Teleports    - deterministic, may chain; sets result.teleported.
    - Confusion    - inverts all movement for the remainder of the current
                     turn AND the entire following turn.
    - Walls        - block movement, increment wall_hits, position unchanged.
    """

    def __init__(self, maze_id: str):
        """
        Args:
            maze_id: 'training' or 'testing'  (passed to maze_loader)
        """
        import maze_loader as ml

        maze_file = "MAZE_1.png" if maze_id == "training" else "MAZE_TEST.png"

        cells, hazards, start, goal_pos = ml.getHMaze(maze_file)
        self.maze = Maze(cells, hazards, start, goal_pos)

        self.maze_id  = maze_id
        self.start_pos: Tuple[int, int] = start.pos

        # -- Episode state --------------------------------------------------
        self._cur_cell          = start
        self._turns_taken       = 0
        self._deaths            = 0
        self._confusions        = 0
        self._cells_explored: set = set()
        self._goal_reached      = False

        # Total executed actions across the episode (drives pit rotation)
        self._total_actions_executed = 0

        # Confusion state: number of *turns* (not actions) remaining where
        # movement is inverted.  Set to 2 when confusion is first triggered
        # (rest-of-current-turn counts as turn 1; next full turn is turn 2).
        self._confused_turns_remaining = 0

    # -- Public API ----------------------------------------------------------

    def reset(self) -> Tuple[int, int]:
        """Reset for a new episode. Returns starting (row, col)."""
        self._cur_cell                 = self.maze.start
        self._turns_taken              = 0
        self._deaths                   = 0
        self._confusions               = 0
        self._cells_explored           = set()
        self._goal_reached             = False
        self._total_actions_executed   = 0
        self._confused_turns_remaining = 0
        return self.start_pos

    def step(self, actions: List[Action]) -> TurnResult:
        """
        Execute one turn (1-5 actions).

        Hazard processing order per action:
          1. Apply confusion inversion if active.
          2. Attempt movement; record wall hit if blocked.
          3. If moved: check cell type and apply hazard immediately.
             - deathpit  -> record death, respawn flag, stop turn.
             - goal       -> record success, stop turn.
             - teleport   -> move to destination (chain if needed).
             - confusion  -> set confused_turns_remaining = 2.
          4. After all actions: decrement confused_turns_remaining if > 0.
          5. Every 5 total executed actions: rotate death pits.
        """
        if not actions:
            raise ValueError("actions list must not be empty")
        if len(actions) > 5:
            raise ValueError("actions list must contain at most 5 actions")

        result = TurnResult()
        result.current_position = self._cur_cell.pos

        currently_confused = self._confused_turns_remaining > 0

        for action in actions:
            # -- 1. Apply confusion inversion ------------------------------
            effective_action = _CONFUSED_MAP[action] if currently_confused else action

            # -- 2. Attempt movement ---------------------------------------
            if effective_action == Action.WAIT:
                result.actions_executed += 1
                self._total_actions_executed += 1
                self._maybe_rotate_pits()
                continue

            attr = _DIR_ATTR[effective_action]
            target = getattr(self._cur_cell, attr)

            if target is None:
                # Wall
                result.wall_hits += 1
                result.actions_executed += 1
                self._total_actions_executed += 1
                self._maybe_rotate_pits()
                continue

            # -- 3. Move succeeded; enter target cell ----------------------
            self._cur_cell = target
            self._cells_explored.add(self._cur_cell.pos)
            result.actions_executed += 1
            self._total_actions_executed += 1

            cell_type = self._cur_cell.type

            # Death pit ----------------------------------------------------
            if cell_type == "deathpit":
                result.is_dead            = True
                result.current_position   = self._cur_cell.pos
                self._deaths             += 1
                # Respawn happens at the START of the next turn (reset cur)
                self._cur_cell            = self.maze.start
                self._maybe_rotate_pits()
                break  # remaining actions ignored

            # Goal ---------------------------------------------------------
            if cell_type == "goal":
                result.is_goal_reached  = True
                result.current_position = self._cur_cell.pos
                self._goal_reached      = True
                self._maybe_rotate_pits()
                break  # episode ends

            # Teleport -----------------------------------------------------
            if cell_type == "teleport":
                result.teleported = True
            # Single jump to destination - do NOT loop, the destination pad
            # also has type "teleport" and points back, so a while loop
            # would bounce back to the source and end up going nowhere.
                if self._cur_cell.tpdest is not None:
                    self._cur_cell = self._cur_cell.tpdest
                    self._cells_explored.add(self._cur_cell.pos)
                # After landing, check if destination is itself hazardous
                post_type = self._cur_cell.type
                if post_type == "deathpit":
                    result.is_dead          = True
                    result.current_position = self._cur_cell.pos
                    self._deaths           += 1
                    self._cur_cell          = self.maze.start
                    self._maybe_rotate_pits()
                    break
                if post_type == "goal":
                    result.is_goal_reached  = True
                    result.current_position = self._cur_cell.pos
                    self._goal_reached      = True
                    self._maybe_rotate_pits()
                    break

            # Confusion ----------------------------------------------------
            if cell_type == "confusion":
                if not result.is_confused:       # only log first hit per turn
                    result.is_confused       = True
                    self._confusions        += 1
                # Invert for: rest of this turn (already handled by flag) +
                # the entire next turn.  Set to 2 so decrement at turn-end
                # leaves 1 for next turn.
                currently_confused               = True
                self._confused_turns_remaining   = 2

            self._maybe_rotate_pits()

        # -- 4. Update confusion counter at end of turn --------------------
        if self._confused_turns_remaining > 0:
            self._confused_turns_remaining -= 1

        # Only update current_position if it wasn't already set by a death or goal
        # (death sets it to the pit pos then moves _cur_cell to start, so we must not overwrite)
        if not result.is_dead and not result.is_goal_reached:
            result.current_position = self._cur_cell.pos
        self._turns_taken += 1
        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken":    self._turns_taken,
            "deaths":         self._deaths,
            "confused":       self._confusions,
            "cells_explored": len(self._cells_explored),
            "goal_reached":   self._goal_reached,
        }

    # -- Internal helpers ----------------------------------------------------

    def _maybe_rotate_pits(self):
        """Rotate death-pit clusters if the 5-action boundary was just crossed."""
        if self._total_actions_executed % 5 == 0 and self._total_actions_executed > 0:
            self.maze.rotate_deathpits()


# ---------------------------------------------
# Utility
# ---------------------------------------------

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
            print(f"  {coord} {hazard_locations[5][coord]}")
    else:
        print("  None")

    print(f"\nConfusion Pads ({len(hazard_locations[6])}):")
    if hazard_locations[6]:
        for coord in sorted(hazard_locations[6]):
            print(f"  {coord}")
    else:
        print("  None")