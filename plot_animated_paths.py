import os
import pickle
import numpy as np
import plot

num_agents = 3
speeds = [1.7, 1.7, 1.7]
budget = [70.0]
num = 35

# with open("data/combined_visit/paths.pkl", "rb") as f:
# with open("data/intro/more/paths.pkl", "rb") as f:
# with open("data/het_speed/paths.pkl", "rb") as f:
# with open("out/maps/video_example.txt/1/paths.pkl", "rb") as f:
with open("out/maps/intro.txt/1/paths.pkl", "rb") as f:
    for i in range(num):
        paths = pickle.load(f)
print(paths)


# with open("data/combined_visit/scores.pkl", "rb") as f:
# with open("data/intro/more/scores.pkl", "rb") as f:
# with open("data/het_speed/scores.pkl", "rb") as f:
# with open("out/maps/video_example.txt/1/scores.pkl", "rb") as f:
with open("out/maps/intro.txt/1/scores.pkl", "rb") as f:
    for i in range(num):
        scores = pickle.load(f)

# with open("maps/paper_example.txt", "r") as f:
with open("maps/intro.txt", "r") as f:
# with open("maps/het_speed.txt", "r") as f:
# with open("maps/video_example.txt", "r") as f:
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

# directory = f"imgs/animations/combined_visit/"
# directory = f"imgs/animations/intro/"
# directory = f"imgs/animations/het_speed/"
directory = f"imgs/animations/video_example_1/"
os.makedirs(directory, exist_ok=True)
plot.plot_animated_paths(
    rpositions,
    rvalues,
    paths,
    scores,
    speeds,
    directory=directory,
    fname="animated_path",
    update_rewards=True,
    side_by_side=True,
)
