import puzzle_utilities as util
import numpy as np
import tkinter as tk

TILE_SIZE = 80  # pixels
ANIM_STEPS = 10  # frames for sliding
ANIM_DELAY = 0.75  # seconds per frame

TILE_SIZE = 80

def draw_puzzle(canvas, state):
    canvas.delete("all")
    arr = state.puzzle_state
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            v = arr[r, c]
            if v != 0:
                x1, y1 = c*TILE_SIZE, r*TILE_SIZE
                x2, y2 = x1+TILE_SIZE, y1+TILE_SIZE
                canvas.create_rectangle(x1, y1, x2, y2, fill="gold", outline="black", width=2)
                canvas.create_text((x1+x2)//2, (y1+y2)//2, text=str(v), font=("Arial", 24, "bold"))
    canvas.update()

def run_viewer(states):
    root = tk.Tk()
    root.title("15 Puzzle Viewer")

    rows, cols = states[0].init_state.shape
    canvas = tk.Canvas(root, width=cols*TILE_SIZE, height=rows*TILE_SIZE, bg="white")
    canvas.pack()

    i = 0  # current index

    def show(idx):
        nonlocal i
        i = idx
        draw_puzzle(canvas, states[i])

    def next_state():
        if i < len(states)-1:
            show(i+1)

    def prev_state():
        if i > 0:
            show(i-1)

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack()

    tk.Button(btn_frame, text="◀ Prev", command=prev_state).pack(side="left")
    tk.Button(btn_frame, text="Next ▶", command=next_state).pack(side="left")

    show(0)
    root.mainloop()

# chatGPT output end
if __name__ == "__main__":
    # TODO: update this code with the new puzzle utilities
    import random

    # Configuration for the demo
    SIDE = 4  # 4x4 puzzle (15-puzzle)
    STEPS = 20  # Number of random steps to scramble

    # Create the solved state: [1, 2, ... 15, 0]
    # puzzle_utilities is imported as util
    home_config = (np.arange(SIDE ** 2) + 1) % (SIDE ** 2)

    try:
        solved_state = util.PuzzleState(home_config)
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