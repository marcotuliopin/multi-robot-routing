import argparse
import os
import pickle
import numpy as np
import plot

parser = argparse.ArgumentParser(description="Plot animated paths.")
parser.add_argument("--speeds", type=float, nargs="+", help="Speed of the agents.")
parser.add_argument("--num", type=int, help="Which solution to plot.")
args = parser.parse_args()


with open("out/maps/1.txt/1/paths.pkl", "rb") as f:
    for i in range(args.num):
        paths = pickle.load(f)
print(paths)


with open("out/maps/1.txt/1/scores.pkl", "rb") as f:
    for i in range(args.num):
        scores = pickle.load(f)


with open("maps/1.txt", "r") as f:
        lines = f.readlines()
        num_rewards = float(lines[0].split(sep=";")[1])
        num_agents = int(lines[1].split(sep=";")[1])
        _, = [float(lines[2].split(sep=";")[1])]
        _ = [1] * int(num_agents)

        rpositions = np.array(
            [list(map(float, line.split(sep=";")[:-1])) for line in lines[3:]]
        )
        rvalues = np.array([float(line.split(";")[2]) for line in lines[3:]])
        rpositions = np.append(rpositions[1:], [rpositions[0]], axis=0)
        rvalues = np.append(rvalues[1:], rvalues[0])


directory = f"imgs/animations/map_1/"
os.makedirs(directory, exist_ok=True)
plot.plot_animated_paths(
    rpositions,
    rvalues,
    paths,
    scores,
    args.speeds,
    directory=directory,
    fname="animated_path",
    update_rewards=True,
    side_by_side=True,
)
