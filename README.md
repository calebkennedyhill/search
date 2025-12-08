# Set up environment
Make sure using a [virtual environment](https://docs.python.org/3.12/library/venv.html)
sudo apt-get install python3-tk 

# Graph search fun: 15-puzzle.

I started with greedy best-first heuristic search. During this first phase I explored a few different heuristics, beginning with the 0-norm (number of misplaced tiles) and 1-norm. These are _bad_ heuristics. Eventually I got my hands on the taxicab norm and the inversion norm, then summed them. That seems to be much better. 

At some point I added in the ability to do A*, then weighted A*. These are a bit slower, but give short solutions (with the right weights).

All this was basically done on the 8-puzzle. For obvious reasons, the 15-puzzle runs much more slowly. With the 8-puzzle and a good heuristic, I was finding solutions with, usually, well under 1000 nodes expanded. This is not the case with the 15-puzzle. I think a fun goal would be getting to a point where I can **solve most 15-puzzles with under 1000 nodes explored**. I'm not sure exaclty whether this would come from just finding better and better heuristics, or from using some fancy other method. It might even be the case that I need to switch from a BFS based search to something like IDA*. 

I'll be back.
