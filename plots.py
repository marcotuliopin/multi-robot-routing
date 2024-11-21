from matplotlib import pyplot as plt
from utils import get_last_valid_idx
import numpy as np


def plot_rewards(ax, reward_p: np.ndarray, reward_value: np.ndarray):
    ax.scatter(reward_p[:, 0], reward_p[:, 1], c="b", s=20, label="Rewards")
    for i, (x, y) in enumerate(reward_p):
        ax.annotate(reward_value[i], (x, y), textcoords="offset points", xytext=(0, 10), ha="center")

    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")


def plot_path(
    ax,
    reward_p: np.ndarray,
    individual: list,
    distmx: np.ndarray,
    budget: int,
    maxdist: float,
    show_radius=False,
    show_path=True,
    color="orange",
):
    total_distance = 0
    curr_reward = 0

    for next_reward in individual:
        if total_distance + distmx[curr_reward, next_reward] > budget:
            break
        start = reward_p[curr_reward]
        end = reward_p[next_reward]
        if show_path:
            ax.plot([start[0], end[0]], [start[1], end[1]], linewidth=2, color=color, marker="o", markersize=4)

        if show_radius:
            circle = plt.Circle((end[0], end[1]), maxdist, color=color, alpha=0.08)
            ax.add_patch(circle)

        total_distance += distmx[curr_reward, next_reward]
        curr_reward = next_reward


def plot_distances(path1: list, path2: list, distmx: np.ndarray, budget: int):
    last_idx = min(get_last_valid_idx(path1, distmx, budget), get_last_valid_idx(path2, distmx, budget))
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = []
    for p1, p2 in zip(path1[:last_idx+1], path2[:last_idx+1]):
        dist.append(distmx[p1, p2])
    ax.plot(range(len(dist)), dist, color='red', marker='o')
    ax.set_ylabel("Distance between agents")
    ax.set_title("Distance between agents in the two paths. Max distance allowed: 3.5")
    plt.grid(True)
    plt.tight_layout()
    plt.show()