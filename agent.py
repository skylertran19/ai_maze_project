from typing import List, Tuple
from enum import Enum
import numpy as np
import maze_environment as me

""" Turn Execution
1. Agent submits action list: [action1, action2, ..., action5]
2. Environment processes each action sequentially
3. Environment returns TurnResult after all actions complete
4. Agent updates internal state and plans next turn
5. Repeat until episode ends
"""

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

