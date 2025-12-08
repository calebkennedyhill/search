import numpy as np
import random

# enter as a vector. reshape and hold as a matrix
class puzzle_state:
    def __init__(self, config):
        conf1 = config
        self.config = conf1.reshape(int(np.sqrt(len(config))),int(np.sqrt(len(config))))
        self.parent = None
        self.d = 0

    def __str__(self):
        pass

    def copy(self):
        state_vec = self.config.copy()
        state_vec = state_vec.reshape( state_vec.shape[0]**2,)
        new = puzzle_state(state_vec)
        new.parent = self.parent
        new.d = self.d
        return new
    
    # chatGPT output
    def __eq__(self, other):
        if not isinstance(other, puzzle_state):
            return False
        return np.array_equal(self.config, other.config)

    def __hash__(self):
        # Convert config to an immutable representation for hashing
        return hash(tuple(self.config.flatten()))
    # chatGPT output end
    
# find the neighbors of a puzzle_state
def moves(par_state: puzzle_state):  
    side_len = par_state.config.shape[0]                       
    a = par_state.config
    i = np.argwhere(a == 0)[0][0]
    j = np.argwhere(a == 0)[0][1]

    # center
    if (i!=0 and i!=side_len-1) and (j!=0 and j!=side_len-1):
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()

        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key

        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key

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
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key

        r.parent = par_state
        d.parent = par_state

        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [r,d]
    elif i==0 and j==side_len-1:
        d = par_state.copy()
        l = par_state.copy()
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key

        l.parent = par_state
        d.parent = par_state

        l.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,d]
    elif i==side_len-1 and j==0:
        u = par_state.copy()
        r = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key

        u.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,r]
    elif i==side_len-1 and j==side_len-1:
        u = par_state.copy()
        l = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key

        l.parent = par_state
        u.parent = par_state

        l.d = par_state.d + 1
        u.d = par_state.d + 1
        return [u,l]

    # non-corner edges
    elif i==0 and(j!=0 and j!=side_len-1):
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        l.parent = par_state
        r.parent = par_state
        d.parent = par_state

        l.d = par_state.d + 1
        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,r,d]
    elif i==side_len-1 and(j!=0 and j!=side_len-1):
        u = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key
        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        l.parent = par_state
        r.parent = par_state
        u.parent = par_state

        l.d = par_state.d + 1
        r.d = par_state.d + 1
        u.d = par_state.d + 1
        return [l,r,u]
    elif (i!=0 and i!=side_len-1) and j==0:
        u = par_state.copy()
        d = par_state.copy()
        r = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        u.parent = par_state
        d.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,d,r]
    elif (i!=0 and i!=side_len-1) and j==side_len-1:
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key
        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        u.parent = par_state
        d.parent = par_state
        l.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        return [u,d,l]
    
def is_found(found_list: list, to_check: puzzle_state):
    for s in found_list:
        if np.array_equal(s.config, to_check.config):
            return True
    
    return False

def manhattan_total(arr):
    sum = 0
    side = arr.shape[0]
    for i in range(side):
        for j in range(side):
            val = arr[i, j]
            if val != 0:
                sum = sum + np.abs( int((val-1)/side) - i) + np.abs( ((val-1)%side) - j)
    return sum

def horz_inv(arr):
    sum = 0
    side = arr.shape[0]
    flat = arr.copy()
    flat = arr.reshape(side**2,)
    for i in range(side**2-1):
        for j in range(i+1,side**2):
            if (flat[i]>flat[j]) and (flat[j]!=0):
                sum = sum + 1
    return sum

def inversion_dist(arr):
    return horz_inv(arr) + horz_inv(arr.transpose())

def init_random_state(home):
    """
    Initializes a random puzzle state from a given home state.

    Generates an initial state by performing a series of random moves on the provided
    home puzzle state and returns the newly generated state. The function ensures
    that the generated state has no parent and resets the depth value.

    Typically home is the solution of the puzzle. It does not have to be.

    Parameters:
    home: puzzle_state
        The initial home state of the puzzle to start from.

    Returns:
    puzzle_state
        A randomly initialized puzzle state derived from the home state.
    """
    here = puzzle_state(home)
    for _ in range(100):
        here = random.choice(moves(here))
    here.parent = None
    here.d = 0
    return here