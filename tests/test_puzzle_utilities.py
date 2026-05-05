import numpy as np
import pytest
from puzzle_utilities import manhattan_total, horz_inv, inversion_dist, Node, moves

def test_manhattan_total():
    # 2x2 solved state
    solved_2x2 = np.array([[1, 2], [3, 0]])
    assert manhattan_total(solved_2x2) == 0
    
    # 2x2 with one swap
    one_swap_2x2 = np.array([[1, 2], [0, 3]])
    # val 3 is at (1, 1), should be at (1, 0). Distance: 1.
    assert manhattan_total(one_swap_2x2) == 1
    
    # 3x3 solved state
    solved_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    assert manhattan_total(solved_3x3) == 0
    
    # 3x3 with one swap
    # Swap 8 and 0
    one_swap_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 0, 8]])
    # val 8 is at (2, 2), should be at (2, 1). Distance: 1.
    assert manhattan_total(one_swap_3x3) == 1

def test_horz_inv():
    # Solved 2x2: [1, 2, 3, 0] -> No inversions
    solved_2x2 = np.array([[1, 2], [3, 0]])
    assert horz_inv(solved_2x2) == 0
    
    # [1, 3, 2, 0] -> 1 inversion (3 > 2)
    inv_2x2 = np.array([[1, 3], [2, 0]])
    assert horz_inv(inv_2x2) == 1
    
    # [3, 2, 1, 0] -> (3, 2), (3, 1), (2, 1) -> 3 inversions
    inv3_2x2 = np.array([[3, 2], [1, 0]])
    assert horz_inv(inv3_2x2) == 3

def test_inversion_dist():
    # Solved 2x2: [[1, 2], [3, 0]]
    # horz_inv: 0
    # transpose: [[1, 3], [2, 0]] -> flat [1, 3, 2, 0] -> 1 inversion (3, 2)
    solved_2x2 = np.array([[1, 2], [3, 0]])
    assert inversion_dist(solved_2x2) == 1
    
    # One inversion in rows, one in cols
    # Row: [[1, 3], [2, 0]] -> (3, 2) -> 1
    # Transpose: [[1, 2], [3, 0]] -> 0
    # Total = 1
    inv_2x2 = np.array([[1, 3], [2, 0]])
    assert inversion_dist(inv_2x2) == 1

def test_node_init():
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    assert node.height == 2
    assert node.width == 2
    assert np.array_equal(node.curr_state.config, arr)
    assert node.is_solved()
    assert node.parent is None
    assert node.d == 0

def test_node_is_solved():
    solved_arr = np.array([[1, 2], [3, 0]])
    not_solved_arr = np.array([[1, 2], [0, 3]])
    
    node1 = Node(2, 2, solved_arr)
    assert node1.is_solved() is True
    
    node2 = Node(2, 2, not_solved_arr)
    assert node2.is_solved() is False

def test_node_copy():
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    node.d = 5
    copy_node = node.copy()
    
    assert copy_node == node
    assert copy_node is not node
    assert copy_node.curr_state is not node.curr_state
    assert np.array_equal(copy_node.curr_state.config, node.curr_state.config)
    assert copy_node.d == 5

def test_node_equality_and_hash():
    arr1 = np.array([[1, 2], [3, 0]])
    arr2 = np.array([[1, 2], [3, 0]])
    arr3 = np.array([[1, 2], [0, 3]])
    
    node1 = Node(2, 2, arr1)
    node2 = Node(2, 2, arr2)
    node3 = Node(2, 2, arr3)
    
    assert node1 == node2
    assert node1 != node3
    assert hash(node1) == hash(node2)
    assert hash(node1) != hash(node3)
    
    # Test with sets
    s = {node1}
    assert node2 in s
    assert node3 not in s

def test_moves_center():
    # 3x3 with blank in center (1, 1)
    # [[1, 2, 3],
    #  [4, 0, 5],
    #  [6, 7, 8]]
    arr = np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
    node = Node(3, 3, arr)
    next_moves = moves(node)
    
    # Should have 4 possible moves: up, down, left, right
    assert len(next_moves) == 4
    for m in next_moves:
        assert m.parent == node
        assert m.d == 1

def test_moves_corner():
    # 2x2 with blank in top-left (0, 0)
    # [[0, 1],
    #  [2, 3]]
    arr = np.array([[0, 1], [2, 3]])
    node = Node(2, 2, arr)
    next_moves = moves(node)
    
    # Should have 2 possible moves: down, right
    assert len(next_moves) == 2

def test_moves_edge():
    # 3x3 with blank in top-edge center (0, 1)
    # [[1, 0, 2],
    #  [3, 4, 5],
    #  [6, 7, 8]]
    arr = np.array([[1, 0, 2], [3, 4, 5], [6, 7, 8]])
    node = Node(3, 3, arr)
    next_moves = moves(node)
    
    # Should have 3 possible moves: left, right, down
    assert len(next_moves) == 3

def test_init_random_state():
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    random_node = node.init_random_state(num_iters=5)
    
    assert isinstance(random_node, Node)
    assert random_node.parent is None
    assert random_node.d == 0
    # It might still be solved by chance, but unlikely for many iters
