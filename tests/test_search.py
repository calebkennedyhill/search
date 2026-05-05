import numpy as np
import pytest
from puzzle_utilities import Node
from w_a_star_variants import get_path, find_sol_and_write_metrics

def test_get_path():
    # Create a small chain of nodes
    n0 = Node(2, 2, np.array([[1, 2], [3, 0]]))
    n1 = Node(2, 2, np.array([[1, 2], [0, 3]]))
    n1.parent = n0
    n1.d = 1
    
    path = get_path(n1)
    assert len(path) == 2
    assert path[0] == n0
    assert path[1] == n1

def test_find_sol_already_solved():
    # 2x2 already solved
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    
    # w1=1.0, w2=1.0 (A*)
    result = find_sol_and_write_metrics(node, "test_output", 100, 1.0, 1.0, write=False)
    
    sol_found, end_node, num_explored, path = result
    assert sol_found is True
    assert end_node == node
    assert num_explored == 1 # Initial state is solved
    assert len(path) == 1
    assert path[0] == node

def test_find_sol_one_move():
    # 2x2 one move away
    # Solved: [[1, 2], [3, 0]]
    # One move: [[1, 2], [0, 3]]
    arr = np.array([[1, 2], [0, 3]])
    node = Node(2, 2, arr)
    
    result = find_sol_and_write_metrics(node, "test_output", 100, 1.0, 1.0, write=False)
    
    sol_found, end_node, num_explored, path = result
    assert sol_found is True
    assert end_node.is_solved()
    assert num_explored == 2 # Initial state + one move
    assert len(path) == 2

def test_find_sol_timeout():
    # 3x3 with very low upper_limit
    arr = np.array([[1, 2, 3], [4, 5, 6], [0, 7, 8]]) # 1 move away
    node = Node(3, 3, arr)
    
    # Limit to 1 node explored
    result = find_sol_and_write_metrics(node, "test_output", 1, 1.0, 1.0, write=False)
    
    assert result == 1 # Timeout returns 1 in the original code
