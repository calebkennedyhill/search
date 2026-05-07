import os
import pandas as pd
import numpy as np
import pytest
from puzzle_utilities import Node
from w_a_star_variants import find_sol_and_write_metrics

def test_find_sol_write_to_file(tmp_path):
    # 2x2 solved state
    arr = np.array([[1, 2], [3, 0]])
    node = Node(2, 2, arr)
    
    output_dir = tmp_path / "variants"
    output_dir.mkdir()
    metrics_path = str(output_dir / "test_run")
    
    # Run with write=True
    result = find_sol_and_write_metrics(node, metrics_path, 100, 1.0, 1.0, write=True)
    
    # Expected filename: test_run_2by2.csv
    expected_file = output_dir / "test_run_2by2.csv"
    assert expected_file.exists()
    
    # Check content
    df = pd.read_csv(expected_file)
    assert len(df) == 1
    assert df.iloc[0]['height'] == 2
    assert df.iloc[0]['width'] == 2
    assert df.iloc[0]['num_states_explored'] == 1
    assert df.iloc[0]['path_length'] == 0

def test_find_sol_timeout_write_to_file(tmp_path):
    # 3x3 one move away, but limit to 1
    arr = np.array([[1, 2, 3], [4, 5, 6], [0, 7, 8]])
    node = Node(3, 3, arr)
    
    output_dir = tmp_path / "variants"
    output_dir.mkdir()
    metrics_path = str(output_dir / "timeout_run")
    
    # Run with write=True and upper_limit=1
    result = find_sol_and_write_metrics(node, metrics_path, 1, 1.0, 1.0, write=True)
    assert result == 1
    
    expected_file = output_dir / "timeout_run_3by3.csv"
    assert expected_file.exists()
    
    df = pd.read_csv(expected_file, header=None)
    
    assert len(df) == 1
    # num_states_explored is 2 because the first node was not the goal and was expanded
    assert df.iloc[0][10] == 2
    assert df.iloc[0][11] == -1 # path_length for timeout
