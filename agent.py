from typing import List, Tuple
from enum import Enum
import numpy as np
import random

from maze_environment import Action, TurnResult, MazeEnvironment


class Agent:
    """
    Q-learning agent for maze navigation.

    State  : (row, col, confused)
    Action : one of the 5 Action enum values
    Q-table: dict { state -> { action -> float } }, persists across episodes.
    Exploration: epsilon-greedy with exponential decay.
    """

    # ------------------------------------------------------------------
    # Hyper-parameters
    # ------------------------------------------------------------------
    ALPHA         = 0.2    # learning rate (higher = faster updates)
    GAMMA         = 0.95   # discount factor (higher = values future rewards more)
    EPSILON_START = 1.0
    EPSILON_MIN   = 0.05
    EPSILON_DECAY = 0.999  # slower decay - need more exploration on a 64x64 maze

    # Rewards - keep it simple, avoid over-shaping
    R_GOAL        =  500.0  # large goal reward
    R_DEATH       =  -50.0  # strong death penalty
    R_STEP        =   -0.5  # small step penalty to encourage efficiency
    R_NEW_CELL    =   +2.0  # reward for visiting a new cell (encourages exploration)
    R_WALL        =   -2.0  # wall hit penalty

    # WAIT is strongly penalised so agent prefers to move
    R_WAIT        =   -5.0

    ACTIONS = [Action.MOVE_UP, Action.MOVE_DOWN,
               Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.WAIT]

    def __init__(self):
        self.memory: dict = {}
        self._known_cells: set = set()

        self._epsilon = self.EPSILON_START

        self.total_episodes      = 0
        self.successful_episodes = 0
        self.total_navigable_cells = 0

        self._episode_history: List[dict] = []

        # Episode-local state
        self._cur_pos:             Tuple[int, int] = (0, 0)
        self._confused:            bool            = False
        self._prev_state                           = None
        self._prev_action:         Action          = None
        self._turn_count:          int             = 0
        self._death_count:         int             = 0
        self._path_length:         int             = 0
        self._unique_visited:      set             = set()
        self._total_cells_visited: int             = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        if last_result is not None:
            self._process_result(last_result)

        state  = self._make_state()
        action = self._choose_action(state)

        self._prev_state  = state
        self._prev_action = action
        self._turn_count += 1

        return [action]

    def reset_episode(self):
        """
        Called at the START of each episode by the training loop.
        Decays epsilon and resets episode-local state only.
        Stats and counters are managed by the training loop.
        """
        self._epsilon = max(self.EPSILON_MIN,
                            self._epsilon * self.EPSILON_DECAY)

        self._cur_pos             = (0, 0)
        self._confused            = False
        self._prev_state          = None
        self._prev_action         = None
        self._turn_count          = 0
        self._death_count         = 0
        self._path_length         = 0
        self._unique_visited      = set()
        self._total_cells_visited = 0

    # ------------------------------------------------------------------
    # Internal: Q-learning update
    # ------------------------------------------------------------------

    def _process_result(self, result: TurnResult):
        new_pos = result.current_position

        # Update confusion state
        self._confused = result.is_confused

        # Track visits (exclude death respawns)
        if not result.is_dead:
            is_new = new_pos not in self._known_cells
            self._path_length         += 1
            self._total_cells_visited += 1
            self._unique_visited.add(new_pos)
            self._known_cells.add(new_pos)
        else:
            is_new = False

        # Compute reward
        if self._prev_action == Action.WAIT:
            reward = self.R_WAIT
        else:
            reward = self.R_STEP

        if result.wall_hits > 0:
            reward += self.R_WALL * result.wall_hits

        if result.is_dead:
            reward += self.R_DEATH
            self._death_count += 1

        if is_new:
            reward += self.R_NEW_CELL  # bonus for discovering new cells

        if result.is_goal_reached:
            reward += self.R_GOAL

        # Q-table update
        if self._prev_state is not None and self._prev_action is not None:
            new_state = self._make_state_from(new_pos, self._confused)
            old_q     = self._get_q(self._prev_state, self._prev_action)
            future_q  = max(self._get_q(new_state, a) for a in self.ACTIONS)
            new_q     = old_q + self.ALPHA * (reward + self.GAMMA * future_q - old_q)
            self._set_q(self._prev_state, self._prev_action, new_q)

        if not result.is_dead:
            self._cur_pos = new_pos

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _choose_action(self, state) -> Action:
        """
        Epsilon-greedy, but never choose WAIT unless it's the only
        positive-Q action. This prevents the agent getting stuck.
        """
        if random.random() < self._epsilon:
            # During exploration, never choose WAIT
            return random.choice([Action.MOVE_UP, Action.MOVE_DOWN,
                                   Action.MOVE_LEFT, Action.MOVE_RIGHT])
        else:
            q_values = {a: self._get_q(state, a) for a in self.ACTIONS}
            # Only allow WAIT if it has a strictly higher Q than all moves
            move_actions = [Action.MOVE_UP, Action.MOVE_DOWN,
                            Action.MOVE_LEFT, Action.MOVE_RIGHT]
            best_move = max(move_actions, key=lambda a: q_values[a])
            if q_values[Action.WAIT] > q_values[best_move]:
                return Action.WAIT
            return best_move

    # ------------------------------------------------------------------
    # State / Q-table helpers
    # ------------------------------------------------------------------

    def _make_state(self):
        return self._make_state_from(self._cur_pos, self._confused)

    def _make_state_from(self, pos: Tuple[int, int], confused: bool):
        return (pos[0], pos[1], int(confused))

    def _get_q(self, state, action: Action) -> float:
        return self.memory.get(state, {}).get(action, 0.0)

    def _set_q(self, state, action: Action, value: float):
        if state not in self.memory:
            self.memory[state] = {}
        self.memory[state][action] = value

    # ------------------------------------------------------------------
    # Episode stats - called by training loop
    # ------------------------------------------------------------------

    def _save_episode_stats(self, goal_reached: bool):
        self._episode_history.append({
            "goal_reached":        goal_reached,
            "turns":               self._turn_count,
            "deaths":              self._death_count,
            "path_length":         self._path_length,
            "unique_cells":        len(self._unique_visited),
            "total_cells_visited": self._total_cells_visited,
        })

    # ------------------------------------------------------------------
    # Grading metrics (Section 8 of spec)
    # ------------------------------------------------------------------

    def success_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return (self.successful_episodes / self.total_episodes) * 100.0

    def avg_path_length(self) -> float:
        successes = [e["path_length"] for e in self._episode_history
                     if e["goal_reached"]]
        return float(np.mean(successes)) if successes else 0.0

    def avg_turns(self) -> float:
        successes = [e["turns"] for e in self._episode_history
                     if e["goal_reached"]]
        return float(np.mean(successes)) if successes else 0.0

    def death_rate(self) -> float:
        total_turns  = sum(e["turns"]  for e in self._episode_history)
        total_deaths = sum(e["deaths"] for e in self._episode_history)
        return (total_deaths / total_turns) if total_turns > 0 else 0.0

    def exploration_efficiency(self) -> float:
        total_unique  = sum(e["unique_cells"]        for e in self._episode_history)
        total_visited = sum(e["total_cells_visited"] for e in self._episode_history)
        return (total_unique / total_visited) if total_visited > 0 else 0.0

    def map_completeness(self) -> float:
        if self.total_navigable_cells == 0:
            return 0.0
        return len(self._known_cells) / self.total_navigable_cells

    def print_stats(self):
        print("--- Agent Performance Metrics ---")
        print(f"  Episodes run      : {self.total_episodes}")
        print(f"  Success rate      : {self.success_rate():.1f}%")
        print(f"  Avg path length   : {self.avg_path_length():.1f}")
        print(f"  Avg turns         : {self.avg_turns():.1f}")
        print(f"  Death rate        : {self.death_rate():.4f}")
        print(f"  Exploration eff.  : {self.exploration_efficiency():.4f}")
        print(f"  Map completeness  : {self.map_completeness():.4f}")
        print(f"  Q-table states    : {len(self.memory)}")
        print(f"  Epsilon           : {self._epsilon:.4f}")