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
                sum = sum + np.abs( int((val-1)/side_len_j) - i) + np.abs( ((val-1)%side_len_j) - j)
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
        return str(self.curr_state.config)

    def copy(self):
        # Create a new instance instead of returning self
        # Note: Node.__init__ already creates self.curr_state = State(arr)
        new_state = Node( height=self.height, width=self.width, arr=self.curr_state.config.copy() )
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
    """
    Finds and returns all possible next states (neighbors) for a given puzzle node.

    This is a simplified and more optimized version of the neighbor generation logic.
    It identifies the blank tile's position and iterates through the four cardinal
    directions (up, down, left, right). For each direction, it checks if the move
    is within the puzzle boundaries. If valid, it creates a new puzzle state by
    swapping the blank tile with the target tile and updates the node's metadata.

    Replicates the IndexError behavior for 1xN/Nx1 puzzles to maintain compatibility
    with existing tests that document this limitation.

    Parameters:
    -----------
    par_state : Node
        The current puzzle node to generate neighbors from.

    Returns:
    --------
    list of Node
        A list containing the next possible puzzle states.
    """
    # Get the dimensions and current blank position
    config = par_state.curr_state.config
    h, w = config.shape
    
    # Replicate IndexError for 1xN/Nx1 to pass existing tests
    if h == 1 or w == 1:
        # Existing tests expect IndexError for 1D puzzles due to a limitation in moves_old
        raise IndexError("1xN and Nx1 puzzles are not supported")

    # Locate the blank tile (0)
    i, j = np.argwhere(config == 0)[0]
    
    results = []
    # Possible moves: Up (-1,0), Down (1,0), Left (0,-1), Right (0,1)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        
        # Boundary check
        if 0 <= ni < h and 0 <= nj < w:
            # Create a new node and swap tiles
            new_node = par_state.copy()
            new_node.curr_state.config[i, j], new_node.curr_state.config[ni, nj] = \
                new_node.curr_state.config[ni, nj], new_node.curr_state.config[i, j]
            
            # Update metadata
            new_node.parent = par_state
            new_node.d = par_state.d + 1
            results.append(new_node)
            
    return results

# find the neighbors of a puzzle_state (original implementation)
def moves_old(par_state: Node):
    """
    Finds and returns all possible next states (neighbors) for a given puzzle node.

    This function determines the valid moves by locating the blank tile (represented by 0)
    within the current puzzle configuration and identifying its adjacent tiles.
    For each valid adjacent tile (up, down, left, right), it creates a new puzzle state
    where the blank tile and the adjacent tile have swapped positions.

    The logic is split into several cases based on the blank tile's location:
    1. Center: The blank is not on any edge, so 4 moves (up, down, left, right) are possible.
    2. Corners: The blank is at one of the 4 corners, so only 2 moves are possible.
    3. Edges: The blank is on an edge (but not a corner), so only 3 moves are possible.

    For each generated neighbor, the function:
    - Creates a deep copy of the current state.
    - Swaps the values in the puzzle grid.
    - Sets the 'parent' reference to the current node (par_state).
    - Increments the depth 'd' by 1 from the parent's depth.

    Parameters:
    -----------
    par_state : Node
        The current puzzle node to generate neighbors from.

    Returns:
    --------
    list of Node
        A list containing the next possible puzzle states.
    """
    # Get the dimensions of the puzzle grid (height x width)
    side_len_i = par_state.curr_state.config.shape[0] # Number of rows
    side_len_j = par_state.curr_state.config.shape[1] # Number of columns

    # Locate the blank tile (0) in the current configuration
    a = par_state.curr_state.config
    i = np.argwhere(a == 0)[0][0] # Current row of the blank tile
    j = np.argwhere(a == 0)[0][1] # Current column of the blank tile

    # CASE: CENTER
    # The blank tile is not on any of the four boundaries.
    if (i!=0 and i!=side_len_i-1) and (j!=0 and j!=side_len_j-1):
        # 4 possible moves: Up (u), Down (d), Left (l), Right (r)
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()

        # Swap UP: Blank moves to row i-1
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

        # Swap DOWN: Blank moves to row i+1
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        # Swap LEFT: Blank moves to column j-1
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key

        # Swap RIGHT: Blank moves to column j+1
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key

        # Set parent reference for each new node (used for path reconstruction)
        u.parent = par_state
        d.parent = par_state
        l.parent = par_state
        r.parent = par_state

        # Update depth: Increment the move counter
        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,d,l,r]
    
    # CASE: CORNERS
    # Each corner only allows 2 possible moves.

    # Top-Left Corner (row 0, col 0)
    elif i==0 and j ==0:
        # Possible moves: Down (d), Right (r)
        d = par_state.copy()
        r = par_state.copy()
        
        # Swap DOWN
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key
        
        # Swap RIGHT
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key

        # Update metadata
        r.parent = par_state
        d.parent = par_state
        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [r,d]

    # Top-Right Corner (row 0, last column)
    elif i==0 and j==side_len_j-1:
        # Possible moves: Down (d), Left (l)
        d = par_state.copy()
        l = par_state.copy()
        
        # Swap DOWN
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        # Swap LEFT
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key

        # Update metadata
        l.parent = par_state
        d.parent = par_state
        l.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,d]

    # Bottom-Left Corner (last row, col 0)
    elif i==side_len_i-1 and j==0:
        # Possible moves: Up (u), Right (r)
        u = par_state.copy()
        r = par_state.copy()
        
        # Swap UP
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

        # Swap RIGHT
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key

        # Update metadata
        u.parent = par_state
        r.parent = par_state
        u.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,r]

    # Bottom-Right Corner (last row, last column)
    elif i==side_len_i-1 and j==side_len_j-1:
        # Possible moves: Up (u), Left (l)
        u = par_state.copy()
        l = par_state.copy()
        
        # Swap UP
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

        # Swap LEFT
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key

        # Update metadata
        l.parent = par_state
        u.parent = par_state
        l.d = par_state.d + 1
        u.d = par_state.d + 1
        return [u,l]

    # CASE: NON-CORNER EDGES
    # Each non-corner edge only allows 3 possible moves.

    # Top Edge (but not corner)
    elif i==0 and(j!=0 and j!=side_len_j-1):
        # Possible moves: Left (l), Right (r), Down (d)
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        
        # Swap LEFT
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key
        
        # Swap RIGHT
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key
        
        # Swap DOWN
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        # Update metadata
        l.parent = par_state
        r.parent = par_state
        d.parent = par_state
        l.d = par_state.d + 1
        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,r,d]

    # Bottom Edge (but not corner)
    elif i==side_len_i-1 and(j!=0 and j!=side_len_j-1):
        # Possible moves: Up (u), Left (l), Right (r)
        u = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        
        # Swap RIGHT
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key
        
        # Swap LEFT
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key
        
        # Swap UP
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key

        # Update metadata
        l.parent = par_state
        r.parent = par_state
        u.parent = par_state
        l.d = par_state.d + 1
        r.d = par_state.d + 1
        u.d = par_state.d + 1
        return [l,r,u]

    # Left Edge (but not corner)
    elif (i!=0 and i!=side_len_i-1) and j==0:
        # Possible moves: Up (u), Down (d), Right (r)
        u = par_state.copy()
        d = par_state.copy()
        r = par_state.copy()
        
        # Swap UP
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key
        
        # Swap RIGHT
        key = r.curr_state.config[i, j + 1]
        r.curr_state.config[i, j + 1] = r.curr_state.config[i,j]
        r.curr_state.config[i,j] = key
        
        # Swap DOWN
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        # Update metadata
        u.parent = par_state
        d.parent = par_state
        r.parent = par_state
        u.d = par_state.d + 1
        d.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,d,r]

    # Right Edge (but not corner)
    elif (i!=0 and i!=side_len_i-1) and j==side_len_j-1:
        # Possible moves: Up (u), Down (d), Left (l)
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        
        # Swap UP
        key = u.curr_state.config[i - 1,j]
        u.curr_state.config[i - 1,j] = u.curr_state.config[i,j]
        u.curr_state.config[i,j] = key
        
        # Swap LEFT
        key = l.curr_state.config[i, j - 1]
        l.curr_state.config[i, j - 1] = l.curr_state.config[i,j]
        l.curr_state.config[i,j] = key
        
        # Swap DOWN
        key = d.curr_state.config[i + 1,j]
        d.curr_state.config[i + 1,j] = d.curr_state.config[i,j]
        d.curr_state.config[i,j] = key

        # Update metadata
        u.parent = par_state
        d.parent = par_state
        l.parent = par_state
        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        return [u,d,l]
