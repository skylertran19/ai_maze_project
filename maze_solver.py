from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import maze_loader as ml
import maze_environment as me

#Uses BFS to find shortest path to exit
def bfs(maze, start, end):

    rows, cols = maze.shape
    queue = deque([start])
    parent = {start: None}
    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr, nc] == 0 and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
    #recreate the path
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    return path[::-1]  # reverse

class testAgent():
    #uses MazeEnvironment
    def __init__(self, environment):
        self.env = environment
    
    def test_move_sequence(self, actions, description="", expected=None):
        """
        Test a sequence of actions and verify TurnResult
        
        Args:
            actions: List of Action objects
            description: Test name
            expected: Dict of expected TurnResult values to verify
                     e.g., {"is_dead": True, "wall_hits": 1, "actions_executed": 2}
        """
        if description:
            print(f"\n{description}")
        print(f"Actions: {[a.name for a in actions]}")
        
        result = self.env.step(actions)
        
        # Print results
        print(f"  Position: {result.current_position}")
        print(f"  Wall hits: {result.wall_hits}")
        print(f"  Actions executed: {result.actions_executed}")
        print(f"  Teleported: {result.teleported}")
        print(f"  Is dead: {result.is_dead}")
        print(f"  Is confused: {result.is_confused}")
        print(f"  Goal reached: {result.is_goal_reached}")
        
        # Validate expectations
        if expected:
            for key, value in expected.items():
                actual = getattr(result, key)
                status = "✓" if actual == value else "✗"
                print(f"  {status} {key}: expected {value}, got {actual}")
        
        return result

def solve_maze(filename):
    maze, original_img = ml.loadmaze(filename)
    rows, cols = maze.shape
    #find entrance and exit
    start_col = np.where(maze[0] == 0)[0][len(np.where(maze[0] == 0)[0]) // 2]
    end_col = np.where(maze[-1] == 0)[0][len(np.where(maze[-1] == 0)[0]) // 2]
    start = (0, int(start_col)) #top
    end = (rows - 1, int(end_col)) #bottom
    print(f"Start: {start}, End: {end}")
    path = bfs(maze, start, end)
    print(f"Path length: {len(path)} pixels")
    # Draw solution
    img_color = original_img.convert("RGB")
    draw = ImageDraw.Draw(img_color)
    path_xy = [(c, r) for r, c in path]
    draw.line(path_xy, fill=(0, 0, 255), width=4)
    # Save output
    output_path = filename.replace(".png", "_bfs.png")
    img_color.save(output_path)
    print(f"Saved to {output_path}")
    
    return maze

if __name__ == "__main__":
    #solves hazardless maze with bfs, check in 1
    m = solve_maze("MAZE_0.png")
    
    # MAZE_1 with MazeEnvironment
    filename = "MAZE_1.png"
    nodes, haz, start_cell, goal_pos = ml.getHMaze(filename)
    base_image_file = filename.replace(".png", "base.png")

    env = me.MazeEnvironment(
        maze_id="testing",
        cells=nodes,
        hazards=haz,
        start_pos=start_cell,
        goal_pos=goal_pos,
        base_image=base_image_file
    )
    
    agent = testAgent(env)
    
    print("TESTING MAZE ENVIRONMENT")
    
    print(f"\nStarting position: {env.reset()}")
    print(f"Start cell pos: {start_cell.pos}")
    
    # Test 1: Move down (valid move)
    agent.test_move_sequence(
        [me.Action.MOVE_DOWN],
        "Test 1: Single valid move DOWN",
        expected={"actions_executed": 1, "wall_hits": 0}
    )
    
    # Test 2: Walk forward 3 steps — dynamically find a valid 3-step path with no wall hits
    env.reset()
    dir_to_action = {"up": me.Action.MOVE_UP, "down": me.Action.MOVE_DOWN,
                     "left": me.Action.MOVE_LEFT, "right": me.Action.MOVE_RIGHT}
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    
    moves = []
    cur = start_cell
    came_from = None
    for _ in range(3):
        forward_dirs = [d for d in ("up","down","left","right")
                        if getattr(cur, d) is not None and d != came_from]
        if not forward_dirs:
            break
        d = forward_dirs[0]
        moves.append(dir_to_action[d])
        came_from = opposite[d]
        cur = getattr(cur, d)
    
    agent.test_move_sequence(
        moves,
        f"Test 2: Multiple moves {[m.name for m in moves]}",
        expected={"actions_executed": 3, "wall_hits": 0}
    )
    
    # Test 3: Move into wall (should increment wall_hits, not change position)
    env.reset()
    agent.test_move_sequence(
        [me.Action.MOVE_UP],
        "Test 3: Move UP from start (should hit wall)",
        expected={"wall_hits": 1, "actions_executed": 1}
    )
    
    # Test 4: Wait action
    agent.test_move_sequence(
        [me.Action.WAIT],
        "Test 4: WAIT action (position should not change)",
        expected={"wall_hits": 0, "actions_executed": 1}
    )
    
    # Test 5: Multiple actions with wall hit in middle
    # State: at (0,31) from Test 3's reset + Test 4's WAIT (no movement)
    # UP hits wall (stay at 0,31) -> DOWN moves to (1,31) -> DOWN hits wall = 2 total wall hits
    agent.test_move_sequence(
        [me.Action.MOVE_UP, me.Action.MOVE_DOWN, me.Action.MOVE_DOWN],
        "Test 5: UP (wall hit), DOWN, DOWN (wall hit)",
        expected={"wall_hits": 2, "actions_executed": 3}
    )
    
    # Test 6: Empty actions should raise error
    print("\n" + "=" * 60)
    print("Test 6: Empty actions list (should raise ValueError)")
    try:
        env.step([])
        print("  ✗ ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 7: Too many actions should raise error
    print("\nTest 7: 6 actions (should raise ValueError)")
    try:
        env.step([me.Action.MOVE_DOWN] * 6)
        print("  ✗ ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

"""
# old test from checkin 2
# MAZE_1
    cells, hazards, start, goal_pos = ml.getHMaze("MAZE_1.png")
    #returns array with all the cells in the maze, 64x64, hazards list for interactions, start node for agent, and goal position
    #starting position for test

    # agent = testAgent(cells[1][17], start) #confusion test

    # agent = testAgent(cells[6][30], start) # teleport test, should end up at 59,55

    agent = testAgent(cells[9][5], start) # deathpit test, should end up at start (0,31)

    print("Start position:", agent.get_pos())

    agent.move_down()
    #agent.move_up() #confusion test only, should move down isntead and end up at 3,17

    print("End position:", agent.get_pos())
"""
