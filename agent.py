from typing import List, Tuple
from enum import Enum
import numpy as np

""" Turn Execution
1. Agent submits action list: [action1, action2, ..., action5]
2. Environment processes each action sequentially
3. Environment returns TurnResult after all actions complete
4. Agent updates internal state and plans next turn
5. Repeat until episode ends
"""

""" Cell class for reference, currently in maze_solver.py
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
"""

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
    maze_id: str
    start_pos : Tuple[int, int]
    def __init__(self, maze_id: str):
        """
        Initialize maze environment
        Args:
        maze_id: 'training' or 'testing'
        """
    def reset(self) -> Tuple[int, int]:
        """
        Reset environment for new episode
        Returns:
        Starting position coordinates
        """
        pass
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
        pass
    def get_episode_stats(self) -> dict:
        """
        Get statistics for current episode
        """
        pass

class Agent:
    """
    Students must implement this interface
    """
    total_navigable_cells = 0
    total_episodes = 0 
    successful_episodes = 0 
    """
    Info needed about a successful episode:
    path_length : number of cells visited (including duplicates), Excludes teleport jumps from count.
    total_turns : number  of turns required to reach goal.
    total_deaths
    unique_cells_discovered
    total_cells_visited

    """

    def __init__(self):
        """
        Initialize agent with empty memory
        """
        self.memory = {} # 64x64, 4,096 cells, store info
    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        """
        Plan next set of actions based on last turn result
        Args:
        last_result: Feedback from previous turn
        (None on first turn of episode)
        Returns:
        List of 1-5 actions to execute
        """
        raise NotImplementedError("Students must implement this method")
    def reset_episode(self):
        """
        Called at start of new episode
        Students can reset episode-specific state
        Memory can be retained for learning
        """
        pass

    #Primary Performance Metrics
    def success_rate(self): #calculated over 5 episodes; total_episodes must = 5
        return (self.successful_episodes / self.total_episodes) * 100
    
    def avg_path_length(self): #lower is better
        # return np.mean() # mean of path_lengths for successful episodes
        pass
    
    def avg_turns(self):
        # avg_turns = mean(turns_taken for successful episodes)
        pass

    def death_rate(self): #over all test episodes
        #death_rate = total_deaths / total_turns
        pass

