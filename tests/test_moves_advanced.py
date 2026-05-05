import numpy as np
import pytest
from puzzle_utilities import Node, moves

def test_moves_2x3():
    # 2x3 puzzle with blank at (0, 0)
    # 0 1 2
    # 3 4 5
    arr = np.array([[0, 1, 2], [3, 4, 5]])
    node = Node(2, 3, arr)
    next_moves = moves(node)
    
    # Possible moves:
    # 1. 3 moves up -> [[3, 1, 2], [0, 4, 5]] (Wait, blank is at (0,0), so we move blank down or right)
    # If 0 is at (0, 0):
    # - Blank down: [[3, 1, 2], [0, 4, 5]] (Wait, blank moves down, so row 1, col 0 becomes 0, and row 0, col 0 becomes 3)
    #   Correct: [[3, 1, 2], [0, 4, 5]]
    # - Blank right: [[1, 0, 2], [3, 4, 5]]
    assert len(next_moves) == 2
    
    # 2x3 puzzle with blank at (0, 1)
    # 1 0 2
    # 3 4 5
    arr = np.array([[1, 0, 2], [3, 4, 5]])
    node = Node(2, 3, arr)
    next_moves = moves(node)
    # Possible: left, right, down
    assert len(next_moves) == 3

def test_moves_3x2():
    # 3x2 puzzle with blank at (1, 1)
    # 1 2
    # 3 0
    # 4 5
    arr = np.array([[1, 2], [3, 0], [4, 5]])
    node = Node(3, 2, arr)
    next_moves = moves(node)
    # 0 is at (1, 1). 
    # Not corner, but it's an edge (right edge).
    # Possible: up, down, left.
    assert len(next_moves) == 3

def test_moves_immutability():
    # Check that moves() does not modify the original node
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    original_config = node.curr_state.config.copy()
    
    next_moves = moves(node)
    assert np.array_equal(node.curr_state.config, original_config)
    for m in next_moves:
        assert m is not node
        assert not np.array_equal(m.curr_state.config, original_config)
