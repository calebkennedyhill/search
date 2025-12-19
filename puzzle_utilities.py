import numpy as np
import random

# enter as a vector. reshape and hold as a matrix
class PuzzleState:
    def __init__(self, num_elements, row_stride:int):
        self.home_1d = (np.arange(num_elements)+1) % num_elements
        self.sol_state = np.array(self.home_1d).reshape(-1, row_stride)
        self.init_state = None
        self.puzzle_state = np.array(self.home_1d).reshape(-1, row_stride)
        self.parent = None
        self.d = 0
        self.row_stride = row_stride

    def init_random_state(self, num_iters: int):
        """
        Initializes a random puzzle state from a given home state aka solution state (sol state).

        Generates an initial state by performing a series of random moves on the provided
        home puzzle state and returns the newly generated state. The function ensures
        that the generated state has no parent and resets the depth value.

        Typically, home is the solution of the puzzle. It does not have to be.

        Parameters:
        home: puzzle_state
            The initial home state of the puzzle to start from.

        Returns:
        puzzle_state
            A randomly initialized puzzle state derived from the home state.
        """
        start = self #TODO: hack fix since moves requires a class, would be a major overhaul
        # start is just the copy of a class which then gets modified.
        start.init_state = start.puzzle_state.copy()  # the moves functions call the copy which depends on init_state which is not fixed currently
        for _ in range(num_iters):
            start = random.choice(moves(start))
        start.parent = None
        start.d = 0
        start.init_state = start.puzzle_state.copy() # the puzzle state gets changed the init_state is kept for tracking
        return start

    def __str__(self):
        pass

    # def copy(self):
    #     # TODO: hack fix just mixing class gen and instance
    #     self.puzzle_state = self.init_state.copy()
    #     # state_vec = self.puzzle_state.copy()
    #     # state_vec = state_vec.reshape( state_vec.shape[0]**2,)
    #     # new = PuzzleState(state_vec)
    #     self.parent = None
    #     self.d = 0
    #     return self

    def copy(self):
        # Create a new instance instead of returning self
        new_state = PuzzleState(len(self.home_1d), self.row_stride)
        new_state.puzzle_state = self.puzzle_state.copy()

        # Copy init_state if it exists
        if self.init_state is not None:
            new_state.init_state = self.init_state.copy()

        new_state.parent = None
        new_state.d = 0
        return new_state
    
    # chatGPT output
    def __eq__(self, other):
        if not isinstance(other, PuzzleState):
            return False
        return np.array_equal(self.puzzle_state, other.puzzle_state)

    def __hash__(self):
        # Convert config to an immutable representation for hashing
        return hash(tuple(self.puzzle_state.flatten()))
    # chatGPT output end
    
# find the neighbors of a puzzle_state
def moves(par_state: PuzzleState):
    side_len_i = par_state.puzzle_state.shape[0]
    side_len_j = par_state.puzzle_state.shape[1] #TODO: so what is the side length with a 2 x 3? I think we need for i another
    # for j
    a = par_state.puzzle_state
    i = np.argwhere(a == 0)[0][0] # row
    j = np.argwhere(a == 0)[0][1] # column

    # center
    if (i!=0 and i!=side_len_i-1) and (j!=0 and j!=side_len_j-1):
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()

        key = u.puzzle_state[i - 1,j]
        u.puzzle_state[i - 1,j] = u.puzzle_state[i,j]
        u.puzzle_state[i,j] = key

        key = d.puzzle_state[i + 1,j]
        d.puzzle_state[i + 1,j] = d.puzzle_state[i,j]
        d.puzzle_state[i,j] = key

        key = l.puzzle_state[i, j - 1]
        l.puzzle_state[i, j - 1] = l.puzzle_state[i,j]
        l.puzzle_state[i,j] = key

        key = r.puzzle_state[i, j + 1]
        r.puzzle_state[i, j + 1] = r.puzzle_state[i,j]
        r.puzzle_state[i,j] = key

        u.parent = par_state
        d.parent = par_state
        l.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,d,l,r]
    
    # corners
    elif i==0 and j ==0:
        d = par_state.copy()
        r = par_state.copy()
        key = d.puzzle_state[i + 1,j]
        d.puzzle_state[i + 1,j] = d.puzzle_state[i,j]
        d.puzzle_state[i,j] = key
        key = r.puzzle_state[i, j + 1]
        r.puzzle_state[i, j + 1] = r.puzzle_state[i,j]
        r.puzzle_state[i,j] = key

        r.parent = par_state
        d.parent = par_state

        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [r,d]
    elif i==0 and j==side_len_j-1:
        d = par_state.copy()
        l = par_state.copy()
        key = d.puzzle_state[i + 1,j]
        d.puzzle_state[i + 1,j] = d.puzzle_state[i,j]
        d.puzzle_state[i,j] = key

        key = l.puzzle_state[i, j - 1]
        l.puzzle_state[i, j - 1] = l.puzzle_state[i,j]
        l.puzzle_state[i,j] = key

        l.parent = par_state
        d.parent = par_state

        l.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,d]
    elif i==side_len_i-1 and j==0:
        u = par_state.copy()
        r = par_state.copy()
        key = u.puzzle_state[i - 1,j]
        u.puzzle_state[i - 1,j] = u.puzzle_state[i,j]
        u.puzzle_state[i,j] = key

        key = r.puzzle_state[i, j + 1]
        r.puzzle_state[i, j + 1] = r.puzzle_state[i,j]
        r.puzzle_state[i,j] = key

        u.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,r]
    elif i==side_len_i-1 and j==side_len_j-1:
        u = par_state.copy()
        l = par_state.copy()
        key = u.puzzle_state[i - 1,j]
        u.puzzle_state[i - 1,j] = u.puzzle_state[i,j]
        u.puzzle_state[i,j] = key

        key = l.puzzle_state[i, j - 1]
        l.puzzle_state[i, j - 1] = l.puzzle_state[i,j]
        l.puzzle_state[i,j] = key

        l.parent = par_state
        u.parent = par_state

        l.d = par_state.d + 1
        u.d = par_state.d + 1
        return [u,l]

    # non-corner edges
    elif i==0 and(j!=0 and j!=side_len_j-1):
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        key = l.puzzle_state[i, j - 1]
        l.puzzle_state[i, j - 1] = l.puzzle_state[i,j]
        l.puzzle_state[i,j] = key
        key = r.puzzle_state[i, j + 1]
        r.puzzle_state[i, j + 1] = r.puzzle_state[i,j]
        r.puzzle_state[i,j] = key
        key = d.puzzle_state[i + 1,j]
        d.puzzle_state[i + 1,j] = d.puzzle_state[i,j]
        d.puzzle_state[i,j] = key

        l.parent = par_state
        r.parent = par_state
        d.parent = par_state

        l.d = par_state.d + 1
        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,r,d]
    elif i==side_len_i-1 and(j!=0 and j!=side_len_j-1):
        u = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        key = r.puzzle_state[i, j + 1]
        r.puzzle_state[i, j + 1] = r.puzzle_state[i,j]
        r.puzzle_state[i,j] = key
        key = l.puzzle_state[i, j - 1]
        l.puzzle_state[i, j - 1] = l.puzzle_state[i,j]
        l.puzzle_state[i,j] = key
        key = u.puzzle_state[i - 1,j]
        u.puzzle_state[i - 1,j] = u.puzzle_state[i,j]
        u.puzzle_state[i,j] = key

        l.parent = par_state
        r.parent = par_state
        u.parent = par_state

        l.d = par_state.d + 1
        r.d = par_state.d + 1
        u.d = par_state.d + 1
        return [l,r,u]
    elif (i!=0 and i!=side_len_i-1) and j==0:
        u = par_state.copy()
        d = par_state.copy()
        r = par_state.copy()
        key = u.puzzle_state[i - 1,j]
        u.puzzle_state[i - 1,j] = u.puzzle_state[i,j]
        u.puzzle_state[i,j] = key
        key = r.puzzle_state[i, j + 1]
        r.puzzle_state[i, j + 1] = r.puzzle_state[i,j]
        r.puzzle_state[i,j] = key
        key = d.puzzle_state[i + 1,j]
        d.puzzle_state[i + 1,j] = d.puzzle_state[i,j]
        d.puzzle_state[i,j] = key

        u.parent = par_state
        d.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,d,r]
    elif (i!=0 and i!=side_len_i-1) and j==side_len_j-1:
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        key = u.puzzle_state[i - 1,j]
        u.puzzle_state[i - 1,j] = u.puzzle_state[i,j]
        u.puzzle_state[i,j] = key
        key = l.puzzle_state[i, j - 1]
        l.puzzle_state[i, j - 1] = l.puzzle_state[i,j]
        l.puzzle_state[i,j] = key
        key = d.puzzle_state[i + 1,j]
        d.puzzle_state[i + 1,j] = d.puzzle_state[i,j]
        d.puzzle_state[i,j] = key

        u.parent = par_state
        d.parent = par_state
        l.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        return [u,d,l]
    
def is_found(found_list: list, to_check: PuzzleState):
    for s in found_list:
        if np.array_equal(s.puzzle_state, to_check.puzzle_state):
            return True
    
    return False

def manhattan_total(arr):
    # Double check if works for nonsquare
    sum = 0
    side_len_i = arr.shape[0]
    side_len_j = arr.shape[1]
    for i in range(side_len_i):
        for j in range(side_len_j):
            val = arr[i, j]
            if val != 0:
                sum = sum + np.abs( int((val-1)/side_len_i) - i) + np.abs( ((val-1)%side_len_j) - j)
    return sum

def horz_inv(arr):
    #TODO: clean this up
    side_len_i = arr.shape[0]
    side_len_j = arr.shape[1]
    num_elem = side_len_i*side_len_j
    flat = arr.flatten()
    sum = 0
    # side = arr.shape[0]
    # flat = arr.copy()
    # flat = arr.reshape(side**2,)
    for i in range(num_elem-1):
        for j in range(i+1,num_elem):
            if (flat[i]>flat[j]) and (flat[j]!=0):
                sum = sum + 1
    return sum

def inversion_dist(arr):
    # doubel check for nonsquare
    return horz_inv(arr) + horz_inv(arr.transpose())