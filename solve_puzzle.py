import numpy as np
from numpy import linalg as la
import pandas as pd
import datetime
import random
import time
import sys
import os

import puzzle_utilities as util
import animate_solution as anim

SIDE = int(sys.argv[1])
TIMEOUT = int(sys.argv[2])
HOME = (np.arange(SIDE**2)+1)%(SIDE**2)


# find the neighbors of a puzzle_state
def moves(par_state: util.puzzle_state):                         
    a = par_state.config
    i = np.argwhere(a == 0)[0][0]
    j = np.argwhere(a == 0)[0][1]

    # center
    if (i!=0 and i!=SIDE-1) and (j!=0 and j!=SIDE-1):
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
    elif i==0 and j==SIDE-1:
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
    elif i==SIDE-1 and j==0:
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
    elif i==SIDE-1 and j==SIDE-1:
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
    elif i==0 and(j!=0 and j!=SIDE-1):
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
    elif i==SIDE-1 and(j!=0 and j!=SIDE-1):
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
    elif (i!=0 and i!=SIDE-1) and j==0:
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
    elif (i!=0 and i!=SIDE-1) and j==SIDE-1:
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

def random_state():
    here = util.puzzle_state(HOME)
    for i in range(100):
        here = random.choice(moves(here))
    here.parent = None
    return here

def is_found(found_list: list, to_check: util.puzzle_state):
    for s in found_list:
        if np.array_equal(s.config, to_check.config):
            return True
    
    return False

def manhattan_total(arr):
    sum = 0
    for i in range(SIDE):
        for j in range(SIDE):
            val = arr[i, j]
            if val != 0:
                sum = sum + np.abs( int((val-1)/SIDE) - i) + np.abs( ((val-1)%SIDE) - j)
    return sum

def heur(s: util.puzzle_state):
    flattened = s.config.copy()
    flattened = flattened.reshape(SIDE**2,)

    return s.d + (util.manhattan_total(s.config) + util.inversion_dist(s.config) )

def get_path(end: util.puzzle_state):
    bwds = [end]
    here = end.copy()

    while here.parent is not None:
        bwds = np.append(bwds, here.parent)
        here = here.parent
    n = len(bwds)
    return np.flip(bwds)

def find_sol(init_state: util.puzzle_state,upper_limit: int):
    qew = [init_state]
    found_states = []
    num_states_explored = 1
    sol_found = False
    flat_init = init_state.config.copy()
    flat_init = flat_init.reshape(SIDE**2,)

    while sol_found is False:
        # pop qew
        here = qew[0]
        found_states = np.append(found_states, here)
        qew = np.delete(qew,0)

        # put nbrs in qew
        nbrs = moves(here)
        for n in nbrs:
            if not is_found(found_states, n):
                qew = np.append(qew, n)

        qew = sorted(qew, key=heur)

        num_states_explored = num_states_explored+1
        if num_states_explored%1000==0:
            print(num_states_explored, 'states explored.')
        if num_states_explored == upper_limit:
            print('--- experiment timeout. ---\n')
            return [sol_found, None, num_states_explored, None]

        flattened = here.config.copy()
        flattened = flattened.reshape(SIDE**2,)

        if int(la.norm(HOME - flattened, 0)) == 0:
            end_time = datetime.datetime.now()
            sol_found = True
            end = here
        
    print('--- Solution found after', num_states_explored, 'states explored. ---')

    p = get_path(end)
    print('Solution length:', len(p),'\n')
    
    output = [sol_found, end, num_states_explored, p]
    
    return output



initial_state = random_state()
sol_found, end, sol_len, solution = find_sol(initial_state, upper_limit=TIMEOUT)

if sol_found:
    anim.run_viewer(solution)

