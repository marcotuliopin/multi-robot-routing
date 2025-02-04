import os
import pickle
import numpy as np
import plot

num_agents = 4
speeds = [1, 1, 1, 1.5]
budget = [150, 150, 150, 200]
num = 14


with open(f"out/front/{num_agents}_agents/{max(budget)}_bgt/paths.pkl", "rb") as f:
    paths = pickle.load(f)
print(paths)

with open(f"out/front/{num_agents}_agents/{max(budget)}_bgt/scores.pkl", "rb") as f:
    scores = pickle.load(f)

with open("maps/dispersed_large.txt", "r") as f:
    lines = f.readlines()
    num_rewards, _ = lines[0].split()
    num_rewards = int(num_rewards)
    rpositions = np.array(
        [list(map(float, line.split())) for line in lines[1 : num_rewards + 1]]
    )
    rvalues = np.array([float(line) for line in lines[num_rewards + 1 :]])

directory = f"imgs/movns/animations/{num_agents}_agents/{max(budget)}_bgt/"
os.makedirs(directory, exist_ok=True)
plot.plot_animated_paths(
    rpositions,
    rvalues,
    paths,
    scores,
    speeds,
    directory=directory,
    fname="animated_path",
    side_by_side=True,
)
