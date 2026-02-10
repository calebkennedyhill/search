import numpy as np
import puzzle_utilities as util
from puzzle_utilities import Node
from numpy import linalg as la
import pandas as pd
import datetime


def get_path(end: Node):
    bwds = [end]
    # here = end.copy() # the copy breaks the animation because the parent get resets to None
    here = end

    while here.parent is not None:
        bwds = np.append(bwds, here.parent)
        here = here.parent
    # n = len(bwds)
    return np.flip(bwds)


def find_sol_and_write_metrics(
        my_ps: Node, path: str, upper_limit: int, w1: float, w2: float, write=False):
    qew = [my_ps]
    found_states = set()
    num_states_explored = 1
    sol_found = False

    print('searching...\t\t\t\tw1 =',w1,'\tw2 =',w2) #\t\tsearch algorithm: wA-star\t\tw1=',w1,'w2=',w2)
    start_time = datetime.datetime.now()
    while not sol_found:
        here, qew = qew[0], qew[1:]
        # qew = np.delete(qew,0)

        # put nbrs in qew
        nbrs = util.moves(here)
        # print(len(nbrs))
        for n in nbrs:
            if n not in found_states:
                found_states.add(n)
                qew = np.append(qew, n)

        # Comment out this line is just BFS,
        # maybe there is a faster way?
        qew = sorted(qew, key=lambda a: a.heur(w1=w1, w2=w2))
        # print("length of qew sorted ",len(qew))

        num_states_explored = num_states_explored+1
        if num_states_explored%100==0:
            print(num_states_explored, 'states explored.')
        if num_states_explored >= upper_limit:
            print('experiment timeout.\n')
            end_time = datetime.datetime.now()
            
            if write:
                metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
                'experiment_start_time': start_time.strftime('%H:%M:%S'),
                'w_1': w1,
                'w_2': w2,
                'height': HEIGHT,
                'width': WIDTH,
                'initial_state': ''.join(np.array2string(my_ps.curr_state.linear(),edgeitems=2)[1:-1].split()),
                'initial_0_norm': int(la.norm( my_ps.sol_state.linear() - my_ps.curr_state.linear(), 0 )),
                'initial_taxi_norm': util.manhattan_total(my_ps.curr_state.config),
                'initial_inv_norm': util.inversion_dist(my_ps.curr_state.config),
                'num_states_explored': num_states_explored,
                'path_length': -1,
                'runtime': end_time-start_time
                }
                df = pd.DataFrame([metrics])
                df.to_csv(path+'_'+str(HEIGHT)+'by'+str(WIDTH)+'.csv', index=False, mode='a', header=False)
            return 1

        if here.is_solved():
            end_time = datetime.datetime.now()
            sol_found = True
            end = here
        
    print('Solution found after', num_states_explored, 'states explored.')

    print('Solution length:', end.d,'\n')

    if write:
        metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
                'experiment_start_time': start_time.strftime('%H:%M:%S'),
                'w_1': w1,
                'w_2': w2,
                'height': HEIGHT,
                'width': WIDTH,
                'initial_state': ''.join(np.array2string(my_ps.curr_state.linear(),edgeitems=2)[1:-1].split()),
                'initial_0_norm': int(la.norm( my_ps.sol_state.linear() - my_ps.curr_state.linear(), 0 )),
                'initial_taxi_norm': util.manhattan_total(my_ps.curr_state.config),
                'initial_inv_norm': util.inversion_dist(my_ps.curr_state.config),
                'num_states_explored': num_states_explored,
                'path_length': end.d,
                'runtime': end_time-start_time
                }
        df = pd.DataFrame([metrics])
        df.to_csv(path+'_'+str(HEIGHT)+'by'+str(WIDTH)+'.csv', index=False, mode='a', header=True)

    return [sol_found, end, num_states_explored, get_path(end)]


if __name__ == '__main__':
    # python w_a-star_variants.py --h 2 --w 2 --maxnode 1000 --num_runs 1 --dir "./variants"
    #to create debug config just edit the config for this script to inlcude the arguments (copy the above --)
    import argparse
    parser = argparse.ArgumentParser(description='Weighted A* variants for puzzle solving')
    parser.add_argument('--h', type=int, help='size of puzzle: height')
    parser.add_argument('--w', type=int, help='size of puzzle: width')
    parser.add_argument('--maxnode', type=int, help='max number of nodes to search')
    parser.add_argument('--num_runs', default=1, type=int, help='Number of runs to perform')
    parser.add_argument('--write', type=bool, default=False, help='True: write results, False: don\'t')
    parser.add_argument('--dir', type=str, default='./variants', help='Directory path for output files')
    args = parser.parse_args()

    MAXNODE = args.maxnode
    NUM_RUNS = args.num_runs
    HEIGHT = args.h
    WIDTH = args.w
    WRITE = args.write
    metrics_path = args.dir
    def make_home():
        return ( (1+np.arange(HEIGHT*WIDTH)) % (HEIGHT*WIDTH)).reshape(HEIGHT,WIDTH) 

    my_puzzle = Node(height=HEIGHT, width=WIDTH, arr=make_home())

    print('\n***********************************************************************************************************************')
    print('Starting', NUM_RUNS, 'runs with puzzle size', HEIGHT, ' by ', WIDTH)
    for i in range(NUM_RUNS):
        initial_state = my_puzzle.init_random_state(num_iters=100)
        print('---------------- Beginning problem number:', i,'----------------\n')
        #### If want to animate
        import animate_solution as anim
        sol_found, end, sol_len, solution=find_sol_and_write_metrics(initial_state, 
                                                                     path=metrics_path, 
                                                                     w1=0.0, w2=1.0, 
                                                                     upper_limit=MAXNODE)
        if sol_found:
            anim.run_viewer(solution)
        find_sol_and_write_metrics(initial_state, path=metrics_path, w1=0.0, w2=1.0, upper_limit=MAXNODE)       # pure heuristic
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=1.0, upper_limit=MAXNODE)       # usual A*
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=2.0, upper_limit=MAXNODE)       # weighted A*
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=1.25, upper_limit=MAXNODE)       # weighted A*
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=1.5, upper_limit=MAXNODE)       # weighted A*

    print('---------------- Experiment finished. ----------------\n\n')


