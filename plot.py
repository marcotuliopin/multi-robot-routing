from matplotlib import pyplot as plt
from utils import interpolate_paths, translate_path_to_coordinates
import numpy as np


def plot_rewards(ax, reward_p: np.ndarray, reward_value: np.ndarray):
    ax.scatter(reward_p[:, 0], reward_p[:, 1], c="b", s=20, label="Rewards")
    for i, (x, y) in enumerate(reward_p):
        ax.annotate(reward_value[i], (x, y), textcoords="offset points", xytext=(0, 10), ha="center")

    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")


def plot_path(
    ax,
    path: list,
    maxdist: float,
    show_radius=False,
    show_path=True,
    color="orange",
):
    prev = path[0]

    for curr in path[1:]:
        if show_path:
            ax.plot([prev[0], curr[0]], [prev[1], curr[1]], linewidth=2, color=color, marker="o", markersize=6)

        if show_radius:
            circle = plt.Circle((curr[0], curr[1]), maxdist, color=color, alpha=0.08)
            ax.add_patch(circle)

        prev = curr


def plot_paths_with_rewards(rpositions, rvalues, individual, MAXD, save_plot=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_rewards(ax, rpositions, rvalues)
    
    individual = translate_path_to_coordinates(individual, rpositions)
    plot_path(ax, individual[0], MAXD, color='orange')
    plot_path(ax, individual[1], MAXD, color='green')

    ax.set_title("Individual Paths")
    plt.grid(True)
    plt.axis('equal')
    plt.ylim(0, None)
    plt.xlim(0, None)

    if save_plot:
        plt.savefig(f'{save_plot}_path.png')

    plt.show()


def plot_distances(path1, path2, positions, max_distance, num_samples=100, save_plot=None):
    interpolated_paths = interpolate_paths(path1, path2, positions, num_samples)
    
    distances = np.linalg.norm(interpolated_paths[0] - interpolated_paths[1], axis=1)
    distances = np.insert(distances, 0, 0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, color='red')
    ax.fill_between(range(len(distances)), distances, color='red', alpha=0.3)
    ax.axhline(max_distance, color="blue", linestyle="--", label=f"Max distance: {max_distance}")
    ax.set_ylabel("Distance between agents")
    ax.set_xlabel("Interpolated steps along paths")
    ax.set_title("Interpolated Distances Between Agents")
    plt.grid(True)
    plt.tight_layout()
    if save_plot:
        plt.savefig(f'{save_plot}_distances.png')
    plt.show()


def plot_objectives(logbook):
    generations = logbook.select("gen")
    max_rewards = logbook.select("max_reward")
    avg_rewards = logbook.select("avg_reward")
    min_distances = logbook.select("min_distance")
    avg_distances = logbook.select("avg_distance")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Recompensa", color="tab:blue")
    ax1.plot(generations, max_rewards, label="Máxima Recompensa", color="tab:blue", linestyle="--")
    ax1.plot(generations, avg_rewards, label="Recompensa Média", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Distância", color="tab:red")
    ax2.plot(generations, min_distances, label="Mínima Distância", color="tab:red", linestyle="--")
    ax2.plot(generations, avg_distances, label="Distância Média", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title("Evolução dos Objetivos")
    plt.show()


def plot_pareto_front(population):
    fitnesses = [ind.fitness.values for ind in population]
    rewards = [fit[0] for fit in fitnesses]
    distances = [fit[1] for fit in fitnesses]

    plt.figure(figsize=(8, 6))
    plt.scatter(distances, rewards, c="blue", label="Soluções")
    plt.xlabel("Distância Mínima (Segundo Objetivo)")
    plt.ylabel("Recompensa Máxima (Primeiro Objetivo)")
    plt.title("Frente de Pareto")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_interpolated_individual(
    individual: list,
    maxdist: float,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_path(ax, individual[0], maxdist, color='orange')
    plot_path(ax, individual[1], maxdist, color='green')

    ax.set_title("Interpolated Individual Paths")
    plt.grid(True)
    plt.axis('equal')
    plt.ylim(0, None)
    plt.xlim(0, None)

    plt.show()