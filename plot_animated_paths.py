import os
import pickle
import numpy as np
import plot

num_agents = 3
speeds = [0.5, 0.5, 0.5]
budget = [70]
num = 31

# with open(f"out/front/{num_agents}_agents/{max(budget)}_bgt/1.0_spd/paths.pkl", "rb") as f:
#     for i in range(num):
#         paths = pickle.load(f)
# print(paths)

with open("paper_example/paths.pkl", "rb") as f:
    paths = pickle.load(f)

# with open(f"out/front/{num_agents}_agents/{max(budget)}_bgt/1.0_spd/scores.pkl", "rb") as f:
#     for i in range(num):
#         scores = pickle.load(f)

with open("paper_example/scores.pkl", "rb") as f:
    scores = pickle.load(f)

with open("maps/paper_example.txt", "r") as f:
    lines = f.readlines()
    num_rewards = lines[0]
    num_rewards = int(num_rewards)
    rpositions = np.array(
        [list(map(float, line.split()[:-1])) for line in lines[1:]]
    )
    rvalues = np.array([float(line.split()[-1]) for line in lines[1:]])

directory = f"imgs/animations/{num_agents}_agents/{max(budget)}_bgt/"
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
