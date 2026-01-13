from ast import literal_eval # for converting strings to integers
import re
import csv
import os
import pandas as pd
import puzzle_utilities as util


if __name__ == "__main__":
    path_to_raw_states = "./korf100.txt"
    path_to_clean_states = "./benchmarks_korf_100.csv"
    with open(path_to_raw_states, "r") as file:
        content = file.read()
        lines = content.split("\n")
        numlines = len(lines)

    for i in range(100):
        ind = 10*i+1
        vals = list(map(literal_eval,lines[ind].split(" ")))
        opt_len = vals.pop(-1)
        state_info = {'initial_state': vals, 'optimal_sol_len': opt_len}
        df = pd.DataFrame([state_info])
        df.to_csv(
            path_to_clean_states, index=False, mode='a', header= (True if i==0 else False)
            )
        
clean = literal_eval(pd.read_csv(path_to_clean_states)['initial_state'][0])

print(clean)