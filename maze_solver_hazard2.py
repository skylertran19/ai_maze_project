from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import math
import maze_loader as ml
from maze_environment import MazeEnvironment, Action
from agent import Agent

# -----------------------------------------------------------------------------
# BFS on raw pixel maze (hazard-free, MAZE_0)
# -----------------------------------------------------------------------------

def bfs_pixels(maze, start, end):
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
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    return path[::-1]

def solve_maze(filename):
    """Naive BFS on raw pixel grid - no hazard awareness. Used for MAZE_0."""
    maze, original_img = ml.loadmaze(filename)
    rows, cols = maze.shape
    start_col = np.where(maze[0] == 0)[0][len(np.where(maze[0] == 0)[0]) // 2]
    end_col   = np.where(maze[-1] == 0)[0][len(np.where(maze[-1] == 0)[0]) // 2]
    start = (0, int(start_col))
    end   = (rows - 1, int(end_col))
    print(f"  Start: {start}, End: {end}")
    path = bfs_pixels(maze, start, end)
    print(f"  Path length: {len(path)} pixels")
    img_color = original_img.convert("RGB")
    draw = ImageDraw.Draw(img_color)
    draw.line([(c, r) for r, c in path], fill=(0, 0, 255), width=4)
    output_path = filename.replace(".png", "_bfs.png")
    img_color.save(output_path)
    print(f"  Saved to {output_path}")
    return maze

# -----------------------------------------------------------------------------
# BFS on cell graph with teleport support (hazard-aware, MAZE_1)
# -----------------------------------------------------------------------------

def bfs_cells(start_cell, goal_pos, hazards):
    """
    BFS on the linked cell graph.
    - Avoids death pits and confusion pads entirely.
    - Treats teleport pads as forced jumps to their destination.
    Returns (path, teleport_jumps).
    """
    avoid = set(hazards[4]) | set(hazards[6])  # death pits + confusion pads
    tp_positions = set(hazards[5].keys())

    queue = deque([start_cell])
    parent = {start_cell.pos: None}
    teleport_jumps = set()

    while queue:
        cur = queue.popleft()
        if cur.pos == goal_pos:
            break
        for d in ["up", "down", "left", "right"]:
            nb = getattr(cur, d)
            if nb is None or nb.pos in parent:
                continue
            if nb.pos in avoid:
                continue  # never step on death pits or confusion pads
            if nb.pos in tp_positions:
                # Stepping on a teleport = forced jump to destination
                dest = nb.tpdest
                if dest is not None and dest.pos not in parent and dest.pos not in avoid:
                    parent[nb.pos] = cur.pos
                    parent[dest.pos] = nb.pos
                    teleport_jumps.add((nb.pos, dest.pos))
                    queue.append(dest)
            else:
                parent[nb.pos] = cur.pos
                queue.append(nb)

    path = []
    pos = goal_pos
    while pos is not None:
        path.append(pos)
        pos = parent.get(pos)
    return path[::-1], teleport_jumps

def solve_maze_hazards(env, filename):
    """
    Optimal BFS solve using the cell graph (respects walls + teleports).
    Draws the path on the original maze image and saves it.
    """
    OFFSET    = 9
    CELL_SIZE = 16

    def cell_to_px(r, c):
        return (OFFSET + c * CELL_SIZE, OFFSET + r * CELL_SIZE)

    def draw_circle(draw, r, c, radius, fill, outline, width=2):
        x, y = cell_to_px(r, c)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                     fill=fill, outline=outline, width=width)

    cells    = env.maze.cells
    hazards  = env.maze.hazards
    start    = env.maze.start
    goal_pos = env.maze.goal_pos

    path, tp_jumps = bfs_cells(start, goal_pos, hazards)
    print(f"  Optimal path length : {len(path)} cells (with teleports)")
    if tp_jumps:
        for jp in tp_jumps:
            print(f"  Teleport jump used  : {jp[0]} -> {jp[1]}")

    img = Image.open(filename).convert("RGBA")

    # Draw path segments - split at each teleport jump
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    odraw   = ImageDraw.Draw(overlay)

    # Build index for quick lookup
    path_index = {pos: i for i, pos in enumerate(path)}
    jump_positions = {jp[0] for jp in tp_jumps} | {jp[1] for jp in tp_jumps}

    # Draw walking segments in blue
    segment = []
    for pos in path:
        if segment and (segment[-1], pos) in tp_jumps:
            # End of a walking segment - draw it
            if len(segment) > 1:
                odraw.line([cell_to_px(r,c) for r,c in segment],
                           fill=(30, 120, 255, 210), width=5)
            segment = [pos]
        else:
            segment.append(pos)
    if len(segment) > 1:
        odraw.line([cell_to_px(r,c) for r,c in segment],
                   fill=(30, 120, 255, 210), width=5)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw teleport jump as orange dashed line
    for jump_from, jump_to in tp_jumps:
        fx, fy = cell_to_px(*jump_from)
        tx, ty = cell_to_px(*jump_to)
        dist  = math.hypot(tx-fx, ty-fy)
        steps = max(int(dist / 10), 2)
        for i in range(steps):
            if i % 2 == 0:
                t0 = i / steps
                t1 = min((i+1) / steps, 1.0)
                x0 = int(fx + (tx-fx)*t0); y0 = int(fy + (ty-fy)*t0)
                x1 = int(fx + (tx-fx)*t1); y1 = int(fy + (ty-fy)*t1)
                draw.line([(x0,y0),(x1,y1)], fill=(255,120,0), width=3)

    # Goal marker on start position
    sr, sc = start.pos
    draw_circle(draw, sr, sc, 12, (220,30,30), (120,0,0))
    sx, sy = cell_to_px(sr, sc)
    draw.text((sx+15, sy-7), "GOAL", fill=(160,0,0))

    # Start marker on goal position
    gr, gc = goal_pos
    draw_circle(draw, gr, gc, 12, (0,210,60), (0,120,0))
    gx, gy = cell_to_px(gr, gc)
    draw.text((gx+15, gy-7), "START", fill=(0,140,0))

    # Teleport pad markers
    for (r, c), color_name in hazards[5].items():
        draw_circle(draw, r, c, 9, (255,130,0), (255,255,255))

    # Death pit markers
    for (r, c) in hazards[4]:
        draw_circle(draw, r, c, 9, (255,50,0), (180,0,0))

    # Confusion pad markers
    for (r, c) in hazards[6]:
        x, y = cell_to_px(r, c)
        draw.rectangle([x-9,y-9,x+9,y+9], fill=(210,190,0), outline=(140,120,0), width=2)

    # Legend removed

    output_path = filename.replace(".png", "_solved.png")
    img.save(output_path)
    print(f"  Saved to {output_path}")
    return path

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _sep(title):
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print('-' * 50)

def _print_result(result):
    print(f"    position  : {result.current_position}")
    print(f"    wall_hits : {result.wall_hits}")
    print(f"    dead      : {result.is_dead}")
    print(f"    confused  : {result.is_confused}")
    print(f"    teleported: {result.teleported}")
    print(f"    goal      : {result.is_goal_reached}")
    print(f"    actions   : {result.actions_executed}")

# -----------------------------------------------------------------------------
# Hazard tests
# -----------------------------------------------------------------------------

def test_wall(env):
    _sep("TEST: Wall")
    env.reset()
    start = env._cur_cell.pos
    result = env.step([Action.MOVE_UP])
    _print_result(result)
    assert result.wall_hits >= 1,             "Expected wall_hits >= 1"
    assert result.current_position == start,  "Position should not change on wall hit"
    print("  PASS")

def test_deathpit(env, cells, hazards):
    _sep("TEST: Death Pit")
    if not hazards[4]:
        print("  SKIP - no death pits in maze")
        return

    pit_pos = next(iter(sorted(hazards[4])))
    r, c = pit_pos

    entry = None
    for dr, dc, direction in [(-1,0,'down'),(1,0,'up'),(0,-1,'right'),(0,1,'left')]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 64 and 0 <= nc < 64:
            neighbour = cells[nr][nc]
            if neighbour.type in ("empty","start") and getattr(neighbour, direction) is not None:
                entry = (neighbour, direction)
                break

    if entry is None:
        print("  SKIP - no walkable neighbour found for pit")
        return

    neighbour_cell, step_dir = entry
    action_map = {'up': Action.MOVE_UP, 'down': Action.MOVE_DOWN,
                  'left': Action.MOVE_LEFT, 'right': Action.MOVE_RIGHT}

    env._cur_cell = neighbour_cell
    print(f"  Standing at {neighbour_cell.pos}, stepping {step_dir} into pit at {pit_pos}")
    result = env.step([action_map[step_dir]])
    _print_result(result)
    assert result.is_dead,                          "Expected is_dead = True"
    assert result.current_position == pit_pos,      "current_position should be the pit"
    assert env._cur_cell.pos == env.start_pos,      "Agent should have respawned at start"
    print("  PASS")

def test_teleport(env, cells, hazards):
    _sep("TEST: Teleport")
    if not hazards[5]:
        print("  SKIP - no teleport pads in maze")
        return

    tp_pos = None
    for pos in sorted(hazards[5].keys()):
        r, c = pos
        if cells[r][c].tpdest is not None:
            tp_pos = pos
            break

    if tp_pos is None:
        print("  SKIP - no teleport pad with a destination found")
        return

    r, c = tp_pos
    expected = cells[r][c].tpdest.pos

    entry = None
    for dr, dc, direction in [(-1,0,'down'),(1,0,'up'),(0,-1,'right'),(0,1,'left')]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 64 and 0 <= nc < 64:
            neighbour = cells[nr][nc]
            if neighbour.type in ("empty","start") and getattr(neighbour, direction) is not None:
                entry = (neighbour, direction)
                break

    if entry is None:
        print("  SKIP - no walkable neighbour found for teleport")
        return

    neighbour_cell, step_dir = entry
    action_map = {'up': Action.MOVE_UP, 'down': Action.MOVE_DOWN,
                  'left': Action.MOVE_LEFT, 'right': Action.MOVE_RIGHT}

    env._cur_cell = neighbour_cell
    print(f"  Standing at {neighbour_cell.pos}, stepping {step_dir} onto teleport at {tp_pos}")
    print(f"  Expected destination: {expected}")
    result = env.step([action_map[step_dir]])
    _print_result(result)
    assert result.teleported,                       "Expected teleported = True"
    assert result.current_position == expected,     f"Expected position {expected}"
    print("  PASS")

def test_confusion(env, cells, hazards):
    _sep("TEST: Confusion")
    if not hazards[6]:
        print("  SKIP - no confusion pads in maze")
        return

    entry = None
    conf_pos = None
    for pos in sorted(hazards[6]):
        r, c = pos
        for dr, dc, direction in [(-1,0,'down'),(1,0,'up'),(0,-1,'right'),(0,1,'left')]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < 64 and 0 <= nc < 64:
                neighbour = cells[nr][nc]
                if neighbour.type in ("empty","start") and getattr(neighbour, direction) is not None:
                    entry = (neighbour, direction)
                    conf_pos = pos
                    break
        if entry:
            break

    if entry is None:
        print("  SKIP - no walkable neighbour found for confusion pad")
        return

    neighbour_cell, step_dir = entry
    action_map = {'up': Action.MOVE_UP, 'down': Action.MOVE_DOWN,
                  'left': Action.MOVE_LEFT, 'right': Action.MOVE_RIGHT}

    env._cur_cell = neighbour_cell
    print(f"  Standing at {neighbour_cell.pos}, stepping {step_dir} onto confusion at {conf_pos}")

    # Turn 1: land on confusion pad
    result1 = env.step([action_map[step_dir]])
    _print_result(result1)
    assert result1.is_confused, "Expected is_confused = True"
    assert env._confused_turns_remaining == 1, \
        f"Expected confused_turns_remaining=1, got {env._confused_turns_remaining}"

    # Turn 2: still confused - MOVE_UP should actually go DOWN
    pos_before = env._cur_cell.pos
    print(f"\n  Turn 2 (confused) - sending MOVE_UP, expect inverted (DOWN)")
    result2 = env.step([Action.MOVE_UP])
    _print_result(result2)
    assert env._confused_turns_remaining == 0, "Confusion should clear after turn 2"
    print(f"  Moved from {pos_before} to {result2.current_position}")

    # Turn 3: back to normal
    pos_before = env._cur_cell.pos
    print(f"\n  Turn 3 (normal) - sending MOVE_UP, expect normal movement")
    result3 = env.step([Action.MOVE_UP])
    _print_result(result3)
    assert env._confused_turns_remaining == 0, "Confusion should still be 0"
    print(f"  Moved from {pos_before} to {result3.current_position}")
    print("  PASS")

def test_pit_rotation(env):
    _sep("TEST: Death Pit Rotation (every 5 actions)")
    if not env.maze._pit_clusters:
        print("  SKIP - no pit clusters detected")
        return

    cluster = env.maze._pit_clusters[0]
    before_dir     = cluster["direction"]
    before_members = set(cluster["members"])
    print(f"  Vertex: {cluster['vertex']}  direction before: {before_dir}")
    print(f"  Members before: {sorted(before_members)}")

    env.reset()
    for _ in range(5):
        env.step([Action.WAIT])

    after_dir     = cluster["direction"]
    after_members = set(cluster["members"])
    print(f"  Direction after : {after_dir}")
    print(f"  Members after   : {sorted(after_members)}")

    if before_dir != -1:
        assert after_dir != before_dir,         "Direction should have changed after 5 actions"
        assert after_members != before_members,  "Members should have shifted after rotation"
    print("  PASS")

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def run_training(num_episodes=50):
    _sep(f"Training Agent ({num_episodes} episodes)")
    env   = MazeEnvironment('training')
    agent = Agent()

    MAX_TURNS = 10000

    for ep in range(num_episodes):
        env.reset()
        agent.reset_episode()
        last_result = None
        goal_reached = False

        while agent._turn_count < MAX_TURNS:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)

            if last_result.is_goal_reached:
                goal_reached = True
                break

        # Save stats and update counters here - we know the true outcome
        agent._save_episode_stats(goal_reached=goal_reached)
        if goal_reached:
            agent.successful_episodes += 1
        agent.total_episodes += 1

        stats = env.get_episode_stats()

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1:4d} | "
                  f"goal: {str(stats['goal_reached']):5} | "
                  f"turns: {stats['turns_taken']:6} | "
                  f"deaths: {stats['deaths']:4} | "
                  f"explored: {stats['cells_explored']:4} | "
                  f"epsilon: {agent._epsilon:.3f}")

    _sep("Final Agent Stats")
    agent.print_stats()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. BFS solve on hazard-free maze (MAZE_0 - no hazards)
    _sep("BFS solve: MAZE_0 (no hazards)")
    solve_maze("MAZE_0.png")

    # 1b. Hazard-aware solve on MAZE_1 (runs after env is built below)

    # 2. Build environment (loads maze internally - use its cells/hazards
    #    so test functions work with the exact same cell objects as the env)
    _sep("Loading MAZE_1 with hazards")
    env = MazeEnvironment('training')
    cells   = env.maze.cells
    hazards = env.maze.hazards
    print(f"  Start: {env.start_pos}  |  Goal: {env.maze.goal_pos}")
    print(f"  Death pits: {len(hazards[4])}, Teleports: {len(hazards[5])}, Confusion: {len(hazards[6])}")

    # 2b. Solve MAZE_1 optimally using cell graph + teleports
    _sep("Solving MAZE_1 (hazard-aware BFS)")
    solve_maze_hazards(env, "MAZE_B_1.png")

    # 3. Hazard tests - confirm each hazard type works correctly
    test_wall(env)
    test_deathpit(env, cells, hazards)
    test_teleport(env, cells, hazards)
    test_confusion(env, cells, hazards)
    test_pit_rotation(env)

    # 5. Train the agent
    run_training(num_episodes=500)