import os
import pickle

import numpy as np

import plot

num_agents = 4
budget = 150
num = 14


with open(f"out/front/{num_agents}_agents/{budget}_bgt/paths.pkl", "rb") as f:
    bounded_path = pickle.load(f)
print(bounded_path)

with open(f"out/front/{num_agents}_agents/{budget}_bgt/scores.pkl", "rb") as f:
    score = pickle.load(f)

with open("maps/dispersed_large.txt", "r") as f:
    lines = f.readlines()
    num_rewards, _ = lines[0].split()
    num_rewards = int(num_rewards)
    rpositions = np.array(
        [list(map(float, line.split())) for line in lines[1 : num_rewards + 1]]
    )
    rvalues = np.array([float(line) for line in lines[num_rewards + 1 :]])

directory = f"imgs/movns/animations/{num_agents}_agents/{budget}_bgt/"
os.makedirs(directory, exist_ok=True)
plot.plot_animated_paths(
    rpositions,
    rvalues,
    bounded_path,
    score,
    directory=directory,
    fname="animated_path",
    side_by_side=True,
)
