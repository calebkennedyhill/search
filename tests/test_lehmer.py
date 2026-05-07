import pytest
from lehmer import factorial, encode, decode

def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(4) == 24
    assert factorial(5) == 120

def test_encode():
    assert encode([]) == 0
    assert encode([0]) == 0
    assert encode([0, 1]) == 0
    assert encode([1, 0]) == 1
    assert encode([0, 1, 2, 3]) == 0
    assert encode([3, 1, 0, 2]) == 20
    assert encode([3, 2, 1, 0]) == 23

def test_decode():
    assert decode(1, 0) == [0]
    assert decode(2, 1) == [1, 0]
    assert decode(3, 5) == [2, 1, 0]
    assert decode(4, 0) == [0, 1, 2, 3]
    assert decode(4, 20) == [3, 1, 0, 2]
    assert decode(4, 23) == [3, 2, 1, 0]
    assert decode(5, 119) == [4, 3, 2, 1, 0]
    assert decode(6, 719) == [5, 4, 3, 2, 1, 0]
    assert decode(7, 5039) == [6, 5, 4, 3, 2, 1, 0]
    assert decode(8, 40319) == [7, 6, 5, 4, 3, 2, 1, 0]
    assert decode(9, 362879) == [8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert decode(10, 3628799) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert decode(11, 39916799) == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert decode(12, 479001599) == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
