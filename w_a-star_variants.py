import numpy as np
import puzzle_utilities as util
from numpy import linalg as la
import pandas as pd
import datetime

def heur(s: util.puzzle_state, w1: float, w2: float):
    # flattened = s.config.copy()
    # flattened = flattened.reshape(SIDE**2,)
    return w1*s.d + w2*(util.manhattan_total(s.config) + util.inversion_dist(s.config) )
    

def get_path(end: util.puzzle_state):
    bwds = [end]
    here = end.copy()

    while here.parent is not None:
        bwds = np.append(bwds, here.parent)
        here = here.parent
    # n = len(bwds)
    return np.flip(bwds)

def find_sol_and_write_metrics(init_state: util.puzzle_state, path: str, upper_limit: int, w1: float, w2: float):
    qew = [init_state]
    # min_heur = heur(init_state, w1=w1, w2=w2)
    found_states = set()
    num_states_explored = 1
    sol_found = False
    flat_init = init_state.config.copy()
    flat_init = flat_init.reshape(SIDE**2,)

    print('searching...\t\t\t\tw1 =',w1,'\tw2 =',w2) #\t\tsearch algorithm: wA-star\t\tw1=',w1,'w2=',w2)
    start_time = datetime.datetime.now()
    while not sol_found:
        here = qew[0]
        qew = np.delete(qew,0)

        # put nbrs in qew
        nbrs = util.moves(here) 
        for n in nbrs:
            if n not in found_states:
                found_states.add(n)
                qew = np.append(qew, n)

        # Comment out this line is just BFS,
        qew = sorted(qew, key=lambda a: heur(a, w1=w1, w2=w2))

        num_states_explored = num_states_explored+1
        if num_states_explored%100==0:
            print(num_states_explored, 'states explored.')
        if num_states_explored >= upper_limit:
            print('experiment timeout.\n')
            end_time = datetime.datetime.now()
            
            metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
               'experiment_start_time': start_time.strftime('%H:%M:%S'),
               'w_1': w1,
               'w_2': w2,
               'size': SIDE,
               'initial_state': ''.join(np.array2string(flat_init,edgeitems=2)[1:-1].split()),
               'initial_0_norm': int(la.norm( HOME - flat_init, 0 )),
               'initial_taxi_norm': util.manhattan_total(init_state.config),
               'initial_inv_norm': util.inversion_dist(init_state.config),
               'num_states_explored': num_states_explored,
               'path_length': -1,
               'runtime': end_time-start_time
               }
            df = pd.DataFrame([metrics])
            df.to_csv(path+'_'+str(SIDE)+'.csv', index=False, mode='a', header=False)
            return 1

        flattened = here.config.copy()
        flattened = flattened.reshape(SIDE**2,)

        if int(la.norm(HOME - flattened, 0)) == 0:
            end_time = datetime.datetime.now()
            sol_found = True
            end = here
        
    print('Solution found after', num_states_explored, 'states explored.')

    print('Solution length:', end.d,'\n')

    metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
               'experiment_start_time': start_time.strftime('%H:%M:%S'),
               'w_1': w1,
               'w_2': w2,
               'size': SIDE,
               'initial_state': ''.join(np.array2string(flat_init,edgeitems=2)[1:-1].split()),
               'initial_0_norm': int(la.norm( HOME - flat_init, 0 )),
               'initial_taxi_norm': util.manhattan_total(init_state.config),
               'initial_inv_norm': util.inversion_dist(init_state.config),
               'num_states_explored': num_states_explored,
               'path_length': end.d,
               'runtime': end_time-start_time
               }
    df = pd.DataFrame([metrics])
    df.to_csv(path+'_'+str(SIDE)+'.csv', index=False, mode='a', header=True)

    return(get_path(end))


if __name__ == '__main__':
    # python w_a-star_variants.py --side 2 --timeout 1000 --num_runs 1 --dir "./variants"
    #TODO: create debugg config so can easily step through
    import argparse
    parser = argparse.ArgumentParser(description='Weighted A* variants for puzzle solving')
    parser.add_argument('--side', type=int, help='Side length of the puzzle')
    parser.add_argument('--timeout', type=int, help='Timeout limit for number of states explored, good number to start 1000')
    parser.add_argument('--num_runs', type=int, help='Number of runs to perform')
    parser.add_argument('--dir', type=str, default='./variants', help='Directory path for output files')
    # metrics_path = '/Users/calebhill/Documents/misc_coding/search/variants'
    args = parser.parse_args()

    SIDE = args.side
    TIMEOUT = args.timeout
    NUM_RUNS = args.num_runs
    HOME = (np.arange(SIDE ** 2) + 1) % (SIDE ** 2)
    metrics_path = args.dir

    print('\n***********************************************************************************************************************')
    print('Starting', NUM_RUNS, 'runs with puzzle size', SIDE)
    for i in range(NUM_RUNS):
        initial_state = util.init_random_state(home=HOME)
        print('---------------- Beginning problem number:', i,'----------------\n')
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=0.0, w2=1.0, upper_limit=TIMEOUT)       # pure heuristic
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=1.0, upper_limit=TIMEOUT)       # usual A*
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=2.0, upper_limit=TIMEOUT)       # weighted A*
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=1.25, upper_limit=TIMEOUT)       # weighted A*
        fresh_copy = initial_state.copy()
        find_sol_and_write_metrics(fresh_copy, path=metrics_path, w1=1.0, w2=1.5, upper_limit=TIMEOUT)       # weighted A*


    print('---------------- Experiment finished. ----------------\n\n')


