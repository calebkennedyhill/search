import numpy as np
import random


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


class State:
    def __init__(self, arr):
        self.config = arr
        pass

    def linear(self):
        return self.config.copy().flatten()

# enter as a vector. reshape and hold as a matrix
class Node:
    def __init__(self, height: int, width: int, arr: np.ndarray):
        assert( arr.shape == (height, width) )
        self.height = height
        self.width = width
        self.curr_state = State( arr )
        self.sol_state = State( ( (1+np.arange(height*width)) % (height*width)).reshape(height,width) )
        self.parent = None
        self.d = 0

    def heur(self, w1: float, w2: float):
        # the heuristic for approximating how close you are to solving the puzzle
        return w1*self.d + w2*(manhattan_total(self.curr_state.config) \
                               + inversion_dist(self.curr_state.config))

    def is_solved(self):
        return np.array_equal(self.curr_state.config, self.sol_state.config)

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
        here = self #TODO: hack fix since moves requires a class, would be a major overhaul
        # start is just the copy of a class which then gets modified.
        here.init_state = here.copy()  
        for _ in range(num_iters):
            here = random.choice(moves(here))
        here.parent = None
        here.d = 0
        return here
    
    def __str__(self):
        pass

    def copy(self):
        # Create a new instance instead of returning self
        new_state = Node( height=self.height, width=self.width, arr=self.curr_state.config.copy() )
        new_state.curr_state = State( self.curr_state.config.copy() )
        new_state.parent = self.parent
        new_state.d = self.d
        return new_state
    
    # chatGPT output
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return np.array_equal(self.curr_state.config, other.curr_state.config)

    def __hash__(self):
        # Convert config to an immutable representation for hashing
        return hash(tuple(self.curr_state.linear()))
    # chatGPT output end
    
# find the neighbors of a puzzle_state
def moves(par_state: Node):
    side_len_i = par_state.curr_state.config.shape[0]
    side_len_j = par_state.curr_state.config.shape[1] 
    # for j
    a = par_state.curr_state.config
    i = np.argwhere(a == 0)[0][0] # row
    j = np.argwhere(a == 0)[0][1] # column

    # center
    if (i!=0 and i!=side_len_i-1) and (j!=0 and j!=side_len_j-1):
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()

        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key

        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key

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
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key

        r.parent = par_state
        d.parent = par_state

        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [r,d]
    elif i==0 and j==side_len_j-1:
        d = par_state.copy()
        l = par_state.copy()
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key

        l.parent = par_state
        d.parent = par_state

        l.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,d]
    elif i==side_len_i-1 and j==0:
        u = par_state.copy()
        r = par_state.copy()
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key

        u.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,r]
    elif i==side_len_i-1 and j==side_len_j-1:
        u = par_state.copy()
        l = par_state.copy()
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key

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
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

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
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

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
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

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
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        u.parent = par_state
        d.parent = par_state
        l.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        return [u,d,l]
