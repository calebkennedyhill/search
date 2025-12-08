import puzzle_utilities as util
import numpy as np

# chatGPT output for visualizing:
import tkinter as tk
import time

TILE_SIZE = 80  # pixels
ANIM_STEPS = 10  # frames for sliding
ANIM_DELAY = 0.75  # seconds per frame

def draw_puzzle(canvas, state: util.puzzle_state, tiles):
    """Draw the puzzle tiles according to arr."""
    arr = state.config
    canvas.delete("all")
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            val = arr[r, c]
            if val != 0:
                x1, y1 = c * TILE_SIZE, r * TILE_SIZE
                x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE
                canvas.create_rectangle(x1, y1, x2, y2, fill="gold", outline="black", width=2)
                canvas.create_text((x1+x2)//2, (y1+y2)//2, text=str(val), font=("Arial", 24, "bold"))
    canvas.update()

def animate_move(canvas, state: util.puzzle_state, start_pos, end_pos):
    """Animate a tile sliding from start_pos to end_pos."""
    arr = state.config

    sr, sc = start_pos
    er, ec = end_pos
    val = arr[sr, sc]
    dx = (ec - sc) * TILE_SIZE / ANIM_STEPS
    dy = (er - sr) * TILE_SIZE / ANIM_STEPS
    
    # Create the moving tile
    x1, y1 = sc * TILE_SIZE, sr * TILE_SIZE
    x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE
    tile = canvas.create_rectangle(x1, y1, x2, y2, fill="gold", outline="black", width=2)
    text = canvas.create_text((x1+x2)//2, (y1+y2)//2, text=str(val), font=("Arial", 24, "bold"))
    
    for _ in range(ANIM_STEPS):
        canvas.move(tile, dx, dy)
        canvas.move(text, dx, dy)
        canvas.update()
        time.sleep(ANIM_DELAY)

def run_animation(states):
    rows, cols = states[0].config.shape
    root = tk.Tk()
    root.title("15 Puzzle Animation")
    canvas = tk.Canvas(root, width=cols*TILE_SIZE, height=rows*TILE_SIZE, bg="white")
    canvas.pack()

    # Draw the first state
    draw_puzzle(canvas, states[0], TILE_SIZE)
    time.sleep(ANIM_DELAY)

    # Animate transitions
    for old, new in zip(states, states[1:]):
        # Find the tile that moved
        diff = np.argwhere(old != new)
        if len(diff) == 2:
            moved_tile_pos = tuple(diff[0]) if old[diff[0][0], diff[0][1]] != 0 else tuple(diff[1])
            empty_pos = tuple(diff[1]) if moved_tile_pos == tuple(diff[0]) else tuple(diff[0])
            animate_move(canvas, old, moved_tile_pos, empty_pos)
        draw_puzzle(canvas, new, TILE_SIZE)
        time.sleep(ANIM_DELAY)

    root.mainloop()

# chatGPT output end
if __name__ == "__main__":
    import random

    # Configuration for the demo
    SIDE = 4  # 4x4 puzzle (15-puzzle)
    STEPS = 20  # Number of random steps to scramble

    # Create the solved state: [1, 2, ... 15, 0]
    # puzzle_utilities is imported as util
    home_config = (np.arange(SIDE ** 2) + 1) % (SIDE ** 2)

    try:
        solved_state = util.puzzle_state(home_config)
    except Exception as e:
        print(f"Error initializing puzzle state: {e}")
        exit(1)

    print(f"Generating a random walk of {STEPS} steps...")

    # Generate a path by walking randomly from the solved state
    # We will record these states to animate them later
    path = [solved_state]
    current_state = solved_state

    for _ in range(STEPS):
        # Get valid moves (neighbors)
        neighbors = util.moves(current_state)
        if not neighbors:
            break

        # Pick a random next state
        next_state = random.choice(neighbors)
        path.append(next_state)
        current_state = next_state

    # To show the puzzle being "solved", we reverse the scramble path
    # (Start at scrambled state -> ... -> Solved state)
    solution_path = list(reversed(path))

    print("Starting animation...")
    run_animation(solution_path)