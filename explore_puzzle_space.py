# idea: BFS explore the whole space and recreate the histograms for # of states with a given norm

# to get some benchmarks: after getting all states, choose some states outside a certain (taxi)^2 + (inv)^2 ellipse as benchmarks
# eccentricity of the ellipse could be mean/sd of norm distribution or something

import numpy as np
import puzzle_utilities as util
import lehmer

from numpy import linalg as la
import pandas as pd
import datetime
import sys

from sklearn.manifold import spectral_embedding
from plotnine import ggplot, aes, geom_point, geom_segment
import pandas as pd

SIDE = int(sys.argv[1])
HOME = (np.arange(SIDE**2)+1)%(SIDE**2)

solved_state = util.puzzle_state(HOME)



# Plan: init (side!) x (side!) matrix M
# For node n generated from parent p,
# mark M[lehmer.encode(n),lehmer.encode(p)]=1 and M[lehmer.encode(p),lehmer.encode(n)]=1
# ...
# This will probably overdo it, but it's a start.

# start with writing down the adj matrix for 100 nodes near the solved state


def explore_and_record():
    frontier = [solved_state]
    discovered = set()
    discovered.add(solved_state)
    num_states_explored = 1
    state_data = []

    nbr_pairs = set()

    print('starting exploration...')
    while len(frontier) != 0 and num_states_explored < 100:
        here = frontier[0]
        frontier = np.delete(frontier,0)

        flattened_here = here.config.reshape(SIDE**2,)

        nbrs = util.moves(here)
        for n in nbrs:
            if n not in discovered:
                discovered.add(n)
                frontier = np.append(frontier,n)
                nbr_pairs.add( (int(lehmer.encode(here.config.flatten())), int(lehmer.encode(n.config.flatten()))) )

        # state_data.append([
        #     num_states_explored, 
        #     la.norm( HOME - flattened_here ,0), 
        #     la.norm( HOME - flattened_here ,1), 
        #     round(la.norm( HOME - flattened_here ,2),3), 
        #     util.manhattan_total(here.config), 
        #     util.inversion_dist(here.config),
        #     ''.join(np.array2string(flattened_here,edgeitems=2)[1:-1].split()),
        #     lehmer.encode(flattened_here)
        #     ])

        num_states_explored = num_states_explored + 1
        if num_states_explored%100 == 0:
            print(num_states_explored,'states explored.')

    # df = pd.DataFrame(columns=['num_explored', 'zero_norm', 'one_norm', 'two_norm', 'taxi_norm', 'inv_norm', 'state_str', 'lehmer_code'],
    #                 data=state_data)
    # df.to_csv(path_or_buf='state_data_'+str(SIDE)+'.csv', index=False)
    print('Done exploring.\n')

    return(nbr_pairs)

coded_nbr_pairs = explore_and_record()
coded_nodes = list(set([elt for tpl in coded_nbr_pairs for elt in tpl]))
num_nodes = len(coded_nodes)

AA = np.full( (num_nodes, num_nodes), fill_value=0, dtype=np.int8)

for n1 in range(num_nodes):
    for n2 in range(num_nodes):
        if ((coded_nodes[n1],coded_nodes[n2]) in coded_nbr_pairs) or ((coded_nodes[n2],coded_nodes[n1]) in coded_nbr_pairs):
            AA[n1,n2] = 1
            AA[n2,n1] = 1

embedding = spectral_embedding(AA, n_components=2)
emb_df = pd.DataFrame({'x': [pt[0] for pt in embedding], 'y': [pt[1] for pt in embedding]})

point_pairs = list([])
for i in range(len(embedding)):
    for j in range(len(embedding)):
        if AA[i,j]==1:
            point_pairs.append([embedding[i], embedding[j]])



x_starts = [pr[0][0] for pr in point_pairs]
y_starts = [pr[0][1] for pr in point_pairs]

x_ends = [pr[1][0] for pr in point_pairs]
y_ends = [pr[1][1] for pr in point_pairs]

# g = ggplot(emb_df) + aes(x='x', y='y') + geom_point()
g = ggplot() + geom_point() + geom_segment(mapping=aes(x=x_starts,y=y_starts, xend=x_ends, yend=y_ends))
print(g)