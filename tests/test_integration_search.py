import numpy as np
import pytest
from puzzle_utilities import Node
from w_a_star_variants import find_sol_and_write_metrics

def test_solve_2x2_complex():
    # 2x2 with several moves
    # Solved: [[1, 2], [3, 0]]
    # 1 2 -> 1 0 -> 0 1 -> 3 1 -> 3 0 -> 1 0 -> 1 2
    # 3 0    3 2    3 2    0 2    1 2    3 2    3 0
    # Let's just use init_random_state and check if it's solvable
    node = Node(2, 2, np.array([[1, 2], [3, 0]]))
    random_node = node.init_random_state(num_iters=10)
    
    result = find_sol_and_write_metrics(random_node, "test_output", 100, 1.0, 1.0, write=False)
    
    assert isinstance(result, list)
    sol_found, end_node, num_explored, path = result
    assert sol_found is True
    assert end_node.is_solved()
    assert len(path) == end_node.d + 1

def test_solve_3x2():
    # 3x2 puzzle
    # Solved:
    # 1 2
    # 3 4
    # 5 0
    node = Node(3, 2, np.array([[1, 2], [3, 4], [5, 0]]))
    # Perform a few manual moves
    # 1 2 -> 1 2 -> 1 0
    # 3 4    3 0    3 2
    # 5 0    5 4    5 4
    arr = np.array([[1, 0], [3, 2], [5, 4]])
    start_node = Node(3, 2, arr)
    
    result = find_sol_and_write_metrics(start_node, "test_output", 100, 1.0, 1.0, write=False)
    assert isinstance(result, list)
    sol_found, end_node, num_explored, path = result
    assert sol_found is True
    assert end_node.is_solved()
    assert end_node.d == 2 # (0, 1) -> (1, 1) -> (2, 1)

def test_solve_3x3_short():
    # 3x3 one move away
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 0, 8]])
    node = Node(3, 3, arr)
    
    result = find_sol_and_write_metrics(node, "test_output", 100, 1.0, 1.0, write=False)
    assert isinstance(result, list)
    sol_found, end_node, num_explored, path = result
    assert sol_found is True
    assert end_node.is_solved()
    assert end_node.d == 1
