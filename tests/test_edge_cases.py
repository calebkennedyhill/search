import numpy as np
import pytest
from puzzle_utilities import Node, State

def test_node_equality_non_node():
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    assert node != "not a node"
    assert node != None

def test_state_linear():
    arr = np.array([[1, 2], [3, 0]])
    state = State(arr)
    linear = state.linear()
    assert np.array_equal(linear, np.array([1, 2, 3, 0]))
    # check copy
    linear[0] = 99
    assert state.config[0, 0] == 1

def test_node_init_invalid_shape():
    arr = np.array([[1, 2, 3]]) # 1x3
    with pytest.raises(AssertionError):
        Node(2, 2, arr)

def test_node_str():
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    s = str(node)
    assert "1" in s
    assert "2" in s
    assert "3" in s
    assert "0" in s
