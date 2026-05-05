import numpy as np
import pytest
from puzzle_utilities import Node, moves

def test_moves_3x3_all_positions():
    """Test all 9 positions of the blank in a 3x3 puzzle."""
    for i in range(3):
        for j in range(3):
            arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 9 is just a placeholder
            arr = arr.reshape(3, 3)
            # Find the value at (i, j) and swap it with 0
            # To make it easier, let's just create the array with 0 at (i, j)
            flat = [1, 2, 3, 4, 5, 6, 7, 8, 0]
            # Replace the element that will be at (i, j) with 0
            val_at_pos = flat[i*3 + j]
            flat[i*3 + j] = 0
            flat[8] = val_at_pos
            
            arr = np.array(flat).reshape(3, 3)
            node = Node(3, 3, arr)
            next_moves = moves(node)
            
            # Corner cases
            if (i == 0 or i == 2) and (j == 0 or j == 2):
                assert len(next_moves) == 2, f"Corner at ({i}, {j}) should have 2 moves, got {len(next_moves)}"
            # Edge cases (not corners)
            elif i == 0 or i == 2 or j == 0 or j == 2:
                assert len(next_moves) == 3, f"Edge at ({i}, {j}) should have 3 moves, got {len(next_moves)}"
            # Center case
            else:
                assert len(next_moves) == 4, f"Center at ({i}, {j}) should have 4 moves, got {len(next_moves)}"
            
            # Verify each move
            for m in next_moves:
                assert m.parent == node
                assert m.d == node.d + 1
                assert m.height == 3
                assert m.width == 3
                # The blank should have moved by exactly one step
                new_blank_pos = np.argwhere(m.curr_state.config == 0)[0]
                dist = np.abs(new_blank_pos[0] - i) + np.abs(new_blank_pos[1] - j)
                assert dist == 1, f"Blank moved from ({i}, {j}) to {new_blank_pos}, distance should be 1"

def test_moves_2x2():
    """Test all positions in a 2x2 puzzle."""
    for i in range(2):
        for j in range(2):
            arr = np.array([[1, 2], [3, 0]])
            val_at_pos = arr[i, j]
            arr[i, j] = 0
            arr[1, 1] = val_at_pos if (i, j) != (1, 1) else 0
            
            node = Node(2, 2, arr)
            next_moves = moves(node)
            assert len(next_moves) == 2, f"Any position in 2x2 should have 2 moves, got {len(next_moves)} at ({i}, {j})"

def test_moves_4x4_center():
    """Test a center position in a 4x4 puzzle."""
    arr = np.array([
        [1, 2, 3, 4],
        [5, 0, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    node = Node(4, 4, arr)
    next_moves = moves(node)
    assert len(next_moves) == 4
    
    # Verify neighbors are correct
    # Move Up: 1 moves down
    up = [m for m in next_moves if np.array_equal(m.curr_state.config, np.array([
        [1, 0, 3, 4],
        [5, 2, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]))]
    assert len(up) == 1
    
    # Move Down: 9 moves up
    down = [m for m in next_moves if np.array_equal(m.curr_state.config, np.array([
        [1, 2, 3, 4],
        [5, 9, 6, 7],
        [8, 0, 10, 11],
        [12, 13, 14, 15]
    ]))]
    assert len(down) == 1
    
    # Move Left: 5 moves right
    left = [m for m in next_moves if np.array_equal(m.curr_state.config, np.array([
        [1, 2, 3, 4],
        [0, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]))]
    assert len(left) == 1
    
    # Move Right: 6 moves left
    right = [m for m in next_moves if np.array_equal(m.curr_state.config, np.array([
        [1, 2, 3, 4],
        [5, 6, 0, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]))]
    assert len(right) == 1

def test_moves_1x3_raises_indexerror():
    """Document current limitation: 1xN puzzles raise IndexError in moves()."""
    arr = np.array([[0, 1, 2]])
    node = Node(1, 3, arr)
    with pytest.raises(IndexError):
        moves(node)

def test_moves_3x1_raises_indexerror():
    """Document current limitation: Nx1 puzzles raise IndexError in moves()."""
    arr = np.array([[0], [1], [2]])
    node = Node(3, 1, arr)
    with pytest.raises(IndexError):
        moves(node)

def test_moves_different_depths():
    """Verify d incrementing works for multiple steps."""
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    node.d = 5
    next_moves = moves(node)
    for m in next_moves:
        assert m.d == 6
        further_moves = moves(m)
        for fm in further_moves:
            assert fm.d == 7

def test_moves_non_square_edges():
    """Test edges in a 2x3 puzzle."""
    # 0 1 2
    # 3 4 5
    arr = np.array([[1, 0, 2], [3, 4, 5]])
    node = Node(2, 3, arr)
    next_moves = moves(node)
    # 0 at (0, 1). This is an edge (Top edge).
    # side_len_i = 2, side_len_j = 3. i = 0, j = 1.
    # Should have: Down, Left, Right (3 moves).
    assert len(next_moves) == 3

    # 1 3 2
    # 4 0 5
    arr = np.array([[1, 3, 2], [4, 0, 5]])
    node = Node(2, 3, arr)
    next_moves = moves(node)
    # 0 at (1, 1). This is an edge (Bottom edge).
    # side_len_i = 2, side_len_j = 3. i = 1, j = 1.
    # Should have: Up, Left, Right (3 moves).
    assert len(next_moves) == 3
