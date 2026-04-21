from collections import defaultdict
import random
import numpy as np
from PIL import Image, ImageDraw
import maze_loader as ml


# =========================
# REWARD CONSTANTS
# =========================
REWARD_GOAL        =  100.0   # reached the exit
REWARD_DEATH_PIT   = -100.0   # fell into a death pit  (episode ends)
REWARD_CONFUSION   =  -10.0   # confusion tile penalty
REWARD_TELEPORT    =   -5.0   # teleport tile (unknown destination, risky)
REWARD_STEP        =   -1.0   # small living penalty so agent prefers short paths


# =========================
# Q-LEARNING CORE
# =========================
def q_learning(maze, start, goal, hazards,
               episodes=5000,
               alpha=0.1,       # learning rate
               gamma=0.95,      # discount factor
               epsilon=1.0,     # starting exploration rate
               epsilon_min=0.05,
               epsilon_decay=0.995):
    """
    Trains a Q-table on the maze and returns:
        q_table  – the learned Q-values
        rewards  – total reward per episode (useful for plotting)
    """

    rows, cols   = maze.shape
    death_pits   = set(hazards[4])
    teleports    = hazards[5]          # dict  {src: dst}
    confusion    = set(hazards[6])
    actions      = [(-1,0),(1,0),(0,-1),(0,1)]   # up, down, left, right

    # Q-table: default 0.0 for every (state, action) pair
    q_table = defaultdict(lambda: [0.0] * len(actions))

    episode_rewards = []

    for ep in range(episodes):
        state        = start
        total_reward = 0.0
        visited_this_episode = set()

        for _ in range(rows * cols * 2):          # step budget per episode
            # ── ε-greedy action selection ──────────────────────────────────
            if random.random() < epsilon:
                action_idx = random.randrange(len(actions))
            else:
                action_idx = int(np.argmax(q_table[state]))

            dr, dc     = actions[action_idx]
            nr, nc     = state[0] + dr, state[1] + dc

            # ── boundary / wall check ─────────────────────────────────────
            if not (0 <= nr < rows and 0 <= nc < cols) or maze[nr][nc] == 1:
                # illegal move → stay in place, small penalty
                reward     = -2.0
                next_state = state
                done       = False

            # ── death pit ─────────────────────────────────────────────────
            elif (nr, nc) in death_pits:
                reward     = REWARD_DEATH_PIT
                next_state = (nr, nc)
                done       = True

            # ── goal ──────────────────────────────────────────────────────
            elif (nr, nc) == goal:
                reward     = REWARD_GOAL
                next_state = (nr, nc)
                done       = True

            # ── teleport ──────────────────────────────────────────────────
            elif (nr, nc) in teleports:
                dest       = teleports[(nr, nc)]
                reward     = REWARD_TELEPORT
                next_state = dest            # agent lands at destination
                done       = False

            # ── confusion zone ────────────────────────────────────────────
            elif (nr, nc) in confusion:
                # In a confusion zone the action taken is randomised
                actual_action = random.choice(actions)
                cr, cc        = nr + actual_action[0], nc + actual_action[1]
                if (0 <= cr < rows and 0 <= cc < cols and maze[cr][cc] != 1):
                    next_state = (cr, cc)
                else:
                    next_state = (nr, nc)    # bumped into wall inside confusion
                reward = REWARD_CONFUSION
                done   = False

            # ── normal move ───────────────────────────────────────────────
            else:
                reward     = REWARD_STEP
                next_state = (nr, nc)
                done       = False

            # ── Q-update (Bellman equation) ───────────────────────────────
            best_next  = max(q_table[next_state])
            old_q      = q_table[state][action_idx]
            q_table[state][action_idx] = (
                old_q + alpha * (reward + gamma * best_next - old_q)
            )

            total_reward += reward
            state         = next_state

            if done:
                break

        # ── decay epsilon after each episode ──────────────────────────────
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-500:])
            print(f"  Episode {ep+1:>5}/{episodes} | "
                  f"ε={epsilon:.3f} | avg reward (last 500): {avg:+.1f}")

    return q_table, episode_rewards


# =========================
# PATH EXTRACTION
# =========================
def extract_path(q_table, maze, start, goal, hazards, max_steps=None):
    """
    Greedily follows the trained Q-table from start → goal.
    Returns the list of (row, col) positions, or an empty list on failure.
    """
    rows, cols  = maze.shape
    max_steps   = max_steps or rows * cols * 2
    death_pits  = set(hazards[4])
    teleports   = hazards[5]
    actions     = [(-1,0),(1,0),(0,-1),(0,1)]

    state  = start
    path   = [state]
    visited = set([state])

    for _ in range(max_steps):
        action_idx = int(np.argmax(q_table[state]))
        dr, dc     = actions[action_idx]
        nr, nc     = state[0] + dr, state[1] + dc

        # boundary / wall → try next-best action
        if not (0 <= nr < rows and 0 <= nc < cols) or maze[nr][nc] == 1:
            sorted_actions = np.argsort(q_table[state])[::-1]
            moved = False
            for idx in sorted_actions[1:]:
                dr2, dc2 = actions[idx]
                nr2, nc2 = state[0] + dr2, state[1] + dc2
                if (0 <= nr2 < rows and 0 <= nc2 < cols and maze[nr2][nc2] != 1):
                    nr, nc = nr2, nc2
                    moved  = True
                    break
            if not moved:
                print("⚠️  Agent is stuck — no valid move from", state)
                break

        next_state = teleports.get((nr, nc), (nr, nc))   # honour teleports
        path.append(next_state)

        if next_state == goal:
            print(f"✅ Goal reached in {len(path)} steps.")
            return path

        if next_state in death_pits:
            print("💀 Agent walked into a death pit during extraction!")
            return path

        if next_state in visited:
            print("🔄 Loop detected during path extraction.")
            break
        visited.add(next_state)
        state = next_state

    print("❌ Could not reach goal — try more training episodes.")
    return path


# =========================
# COORDINATE NORMALISER
# =========================
def to_tuple(pos):
    """
    Ensure a position is a plain (int, int) tuple regardless of how
    maze_loader returns it — handles:
        • already a tuple/list of ints  → (r, c)
        • string  "(r, c)"              → (r, c)
        • object with .pos attribute    → recurse on .pos
        • object with .r/.c attributes  → (r, c)
    """
    if hasattr(pos, "pos"):
        return to_tuple(pos.pos)
    if hasattr(pos, "r") and hasattr(pos, "c"):
        return (int(pos.r), int(pos.c))
    if isinstance(pos, (list, tuple)):
        return (int(pos[0]), int(pos[1]))
    if isinstance(pos, str):
        # handles "(3, 5)" or "3,5" or "3 5"
        nums = [s for s in pos.replace("(","").replace(")","").replace(","," ").split() if s.lstrip("-").isdigit()]
        return (int(nums[0]), int(nums[1]))
    raise TypeError(f"Cannot convert {type(pos)} to (row, col) tuple: {pos!r}")


# =========================
# SCALE FUNCTION
# =========================
def scale_to_image(coord, img_size, grid_size):
    r, c = coord
    h, w = img_size
    gh, gw = grid_size
    x = int(c * w / gw)
    y = int(r * h / gh)
    return (x, y)


# =========================
# DRAW FUNCTION
# =========================
def draw_maze(img_path, path, death_pits, teleports, confusion, grid_shape):
    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    gh, gw = grid_shape

    # PATH
    if len(path) >= 2:
        path_pixels = [scale_to_image(p, (w, h), (gh, gw)) for p in path]
        draw.line(path_pixels, fill=(0, 0, 255), width=3)

    # DEATH PITS
    for r, c in death_pits:
        x, y = scale_to_image((r, c), (w, h), (gh, gw))
        draw.rectangle([(x-2, y-2), (x+2, y+2)], fill=(255, 0, 0))

    # CONFUSION ZONES
    for r, c in confusion:
        x, y = scale_to_image((r, c), (w, h), (gh, gw))
        draw.rectangle([(x-2, y-2), (x+2, y+2)], fill=(255, 255, 0))

    # TELEPORTS
    for r, c in teleports:
        x, y = scale_to_image((r, c), (w, h), (gh, gw))
        draw.rectangle([(x-3, y-3), (x+3, y+3)], outline=(0, 255, 0), width=2)

    out = img_path.replace(".png", "_QSOLVED.png")
    img.save(out)
    img.show()
    print("🖼️  Saved:", out)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # ── Load maze ─────────────────────────────────────────────────────────
    maze, hazards, start, goal = ml.getHMaze("MAZE_1.png")

    # Normalise start / goal to plain (int, int) tuples
    start = to_tuple(start)
    goal  = to_tuple(goal)

    # Normalise every coordinate inside the hazard dict
    death_pits_norm = set(to_tuple(p) for p in hazards[4])
    confusion_norm  = set(to_tuple(p) for p in hazards[6])

    # hazards[5] is {coord: colour_string} — pair same-colour pads bidirectionally
    # e.g. {(7,30):'orange', (59,55):'orange'} → (7,30)↔(59,55)
    from collections import defaultdict
    colour_groups = defaultdict(list)
    for coord, colour in hazards[5].items():
        colour_groups[colour].append(to_tuple(coord))

    teleports_norm = {}
    for colour, pads in colour_groups.items():
        if len(pads) == 2:
            a, b = pads
            teleports_norm[a] = b   # stepping on a → lands on b
            teleports_norm[b] = a   # stepping on b → lands on a
        else:
            # odd number of pads — skip or loop to self
            print(f"⚠️  Colour '{colour}' has {len(pads)} pad(s), skipping teleport pairing.")

    # Rebuild hazards with normalised coords so q_learning receives clean data
    hazards = dict(hazards)
    hazards[4] = death_pits_norm
    hazards[5] = teleports_norm
    hazards[6] = confusion_norm

    death_pits = death_pits_norm
    confusion  = confusion_norm
    teleports  = list(teleports_norm.keys())

    print(f"Start : {start}")
    print(f"Goal  : {goal}")
    print(f"Grid  : {maze.shape}")
    print(f"Death pits : {len(death_pits)}")
    print(f"Confusion  : {len(confusion)}")
    print(f"Teleports  : {len(teleports)}")
    print()

    # ── Train Q-table ─────────────────────────────────────────────────────
    print("🧠 Training Q-learning agent …")
    q_table, rewards = q_learning(
        maze, start, goal, hazards,
        episodes     = 5000,
        alpha        = 0.1,
        gamma        = 0.95,
        epsilon      = 1.0,
        epsilon_min  = 0.05,
        epsilon_decay= 0.995,
    )
    print(f"\nTraining complete. Final avg reward: {np.mean(rewards[-500:]):+.1f}\n")

    # ── Extract greedy path ────────────────────────────────────────────────
    print("🔍 Extracting greedy path …")
    path = extract_path(q_table, maze, start, goal, hazards)
    print(f"Path length: {len(path)}")
    for i, step in enumerate(path):
        print(f"Step {i:>3}: {step}")

    # ── Visualise ─────────────────────────────────────────────────────────
    draw_maze("MAZE_1.png", path, death_pits, teleports, confusion, maze.shape)