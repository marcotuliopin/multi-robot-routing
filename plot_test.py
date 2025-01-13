import pickle

import numpy as np

import plot


with open('out/bounded_path_17.pkl', 'rb') as f:
    bounded_path = pickle.load(f)
    score = pickle.load(f)

print(bounded_path)
print(score)

with open('maps/dispersed_large.txt', "r") as f:
    lines = f.readlines()
    num_rewards = int(lines[0])
    rpositions = np.array(
        [list(map(float, line.split())) for line in lines[1 : num_rewards + 1]]
    )
    rvalues = np.array([float(line) for line in lines[num_rewards + 1 :]])

plot.plot_animated_paths(rpositions, rvalues, bounded_path, score, 4, directory='imgs/movns/movns', fname='animated_path')