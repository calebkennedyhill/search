#TODO: update this code with the new puzzle utilities

# idea: BFS explore the whole space and recreate the histograms for # of states with a given norm

# to get some benchmarks: after getting all states, choose some states outside a certain (taxi)^2 + (inv)^2 ellipse as benchmarks
# eccentricity of the ellipse could be mean/sd of norm distribution or something

import numpy as np
import puzzle_utilities as util
from puzzle_utilities import Node
from numpy import linalg as la
import pandas as pd
import datetime

from matplotlib import pyplot as plt
from sklearn.manifold import spectral_embedding
import lehmer


#               CUT
# SIDE = int(sys.argv[1])
# LIMIT = int(sys.argv[2])
# HOME = (np.arange(SIDE**2)+1)%(SIDE**2) # change for 2x3

# solved_state = util.PuzzleState(HOME)
#               CUT


def explore_and_record_nbrs(upper_limit: int):

    def make_home():
        return ( (1+np.arange(HEIGHT*WIDTH)) % (HEIGHT*WIDTH)).reshape(HEIGHT,WIDTH) 

    solved_state = Node(height=HEIGHT, width=WIDTH, arr=make_home())

    frontier = [solved_state]
    discovered = set()
    discovered.add(solved_state)
    num_states_explored = 1
    state_data = []

    nbr_pairs = set()

    print('starting exploration...')
    while len(frontier) != 0 and num_states_explored < upper_limit:
        here, frontier = frontier[0], frontier[1:]

        nbrs = util.moves(here)
        for n in nbrs:
            nbr_pairs.add(
                (int(lehmer.encode(here.curr_state.linear())), int(lehmer.encode(n.curr_state.linear())))
                )
            if n not in discovered:
                discovered.add(n)
                frontier = np.append(frontier, n)

        num_states_explored = num_states_explored + 1
        if num_states_explored%100 == 0:
            print(num_states_explored,'states explored.')


    print('Done exploring.\n')

    return(nbr_pairs)









if __name__ == "__main__":

    # python explore_puzzle_space.py --h 2 --w 3 --maxnode 1000
    #to create debug config just edit the config for this script to inlcude the arguments (copy the above --)
    import argparse
    parser = argparse.ArgumentParser(description='Weighted A* variants for puzzle solving')
    parser.add_argument('--h', type=int, help='size of puzzle: height')
    parser.add_argument('--w', type=int, help='size of puzzle: width')
    parser.add_argument('--maxnode', type=int, help='max number of nodes to search')
    args = parser.parse_args()

    MAXNODE = args.maxnode
    HEIGHT = args.h
    WIDTH = args.w

    coded_nbr_pairs = explore_and_record_nbrs(upper_limit=MAXNODE)
    coded_nodes = list(set([elt for tpl in coded_nbr_pairs for elt in tpl]))
    num_nodes = len(coded_nodes)

    AA = np.full( (num_nodes, num_nodes), fill_value=0, dtype=np.int8)

    for n1 in range(num_nodes):
        for n2 in range(num_nodes):
            if ((coded_nodes[n1],coded_nodes[n2]) in coded_nbr_pairs) or ((coded_nodes[n2],coded_nodes[n1]) in coded_nbr_pairs):
                AA[n1,n2] = 1
                AA[n2,n1] = 1

    embedding = spectral_embedding(AA, n_components=3, random_state=42)
    rows, cols = np.where(AA == 1)
    point_pairs = [(embedding[i], embedding[j]) for i, j in zip(rows,cols)]

    xs, ys, zs = [], [], []
    for p, q in point_pairs:
        xs += [p[0], q[0], np.nan]
        ys += [p[1], q[1], np.nan]
        zs += [p[2], q[2], np.nan]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs,ys,zs)

    ax.scatter([pt[0] for pt in embedding], 
               [pt[1] for pt in embedding], 
               [pt[2] for pt in embedding], 
               s=10, color="black")
    ax.set_axis_off()

    plt.show()