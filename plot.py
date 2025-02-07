import os
from matplotlib import animation, pyplot as plt
from utils import (
    calculate_rssi_history,
    interpolate_paths,
    interpolate_paths_with_speeds,
    translate_path_to_coordinates,
    calculate_rssi,
)
import numpy as np


def plot_rewards(
    ax, reward_p: np.ndarray, reward_value: np.ndarray, not_plot: set = set()
):
    """
    Plots the rewards on the given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot on.
    reward_p (np.ndarray): The positions of the rewards.
    reward_value (np.ndarray): The values of the rewards.
    not_plot (set): The set of rewards to not plot.
    """
    rewards_to_plot = [i for i in range(len(reward_p)) if i not in not_plot]
    ax.scatter(
        reward_p[rewards_to_plot, 0],
        reward_p[rewards_to_plot, 1],
        c="b",
        s=20,
        label="Rewards",
    )
    for i, (x, y) in enumerate(reward_p):
        if i in not_plot:
            continue
        ax.annotate(
            reward_value[i],
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")


def plot_path(
    ax,
    path: list,
    show_path=True,
    color="orange",
    show_arrow=True,
):
    """
    Plots a path on the given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot on.
    path (list): The path to plot.
    maxdist (float): The maximum distance for the radius.
    show_radius (bool): Whether to show the radius.
    show_path (bool): Whether to show the path.
    color (str): The color of the path.
    show_arrow (bool): Whether to show arrows on the path.
    """
    prev = path[0]

    for curr in path[1:]:
        if show_path:
            ax.plot([prev[0], curr[0]], [prev[1], curr[1]], linewidth=2, color=color)

            dx, dy = curr[0] - prev[0], curr[1] - prev[1]

            if show_arrow:
                ax.quiver(
                    prev[0],
                    prev[1],
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                )

        prev = curr


def get_path_length(path):
    """
    Calculates the length of a path.

    Parameters:
    path (list): The path to calculate the length of.

    Returns:
    float: The length of the path.
    """
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def plot_paths_with_rewards(
    rpositions, rvalues, individual, scores, directory=None, fname=None
):
    """
    Plots the paths with rewards.

    Parameters:
    rpositions (np.ndarray): The positions of the rewards.
    rvalues (np.ndarray): The values of the rewards.
    individual (list): The paths of the individual.
    scores (list): The scores of the individual.
    MAXD (float): The maximum distance for the radius.
    directory (str): The directory to save the plot.
    fname (str): The filename to save the plot.
    """
    n_agents = len(individual)
    colormap = plt.cm.get_cmap("tab10", n_agents)
    colors = [colormap(i) for i in range(n_agents)]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("grey")
    ax.set_facecolor("grey")
    plot_rewards(ax, rpositions, rvalues)

    individual = translate_path_to_coordinates(individual, rpositions)
    length = [get_path_length(ind) for ind in individual]

    # Plot the paths of the agents.
    for i in range(n_agents):
        plot_path(ax, individual[i], color=colors[i])

    ax.set_title(
        "Individual Paths - score: "
        + str([int(score) for score in scores])
        + "- length: "
        + str([int(l) for l in length])
    )
    plt.grid(True)
    plt.axis("equal")
    plt.ylim(0, None)
    plt.xlim(0, None)

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/{fname if fname else 'paths'}.png")
    plt.close()


def get_collection_history(interpolation, individual, coordinates, rvalues):
    """
    Gets the collection history of the rewards.

    Parameters:
    interpolation (list): The interpolated paths.
    individual (list): The paths of the individual.
    coordinates (np.ndarray): The coordinates of the rewards.
    rvalues (np.ndarray): The values of the rewards.

    Returns:
    tuple: The collected rewards, collected values, and collection indices.
    """
    collected_rewards = [set()]
    collected_values = [0]
    path_idx = [0] * len(individual)
    collection_idx = []

    for i in range(len(interpolation[0])):
        for j, idx in enumerate(path_idx):
            if idx >= len(individual[j]):
                continue
            if all(
                abs(coordinates[j][idx][z] - interpolation[j][i][z]) < 1
                for z in range(2)
            ):
                collected_rewards.append(
                    collected_rewards[-1].union({individual[j][idx]})
                )
                collected_values.append(
                    collected_values[-1] + rvalues[individual[j][idx]]
                )
                collection_idx.append(i)
                path_idx[j] += 1

    return collected_rewards, collected_values, collection_idx


def plot_animated_paths(
    rpositions,
    rvalues,
    paths,
    scores,
    speeds,
    directory=None,
    fname=None,
    side_by_side=False,
):
    """
    Plots the animated paths.

    Parameters:
    rpositions (np.ndarray): The positions of the rewards.
    rvalues (np.ndarray): The values of the rewards.
    paths (list): The paths of the solution.
    scores (list): The scores of the solution.
    speeds (list): The speeds of the agents.
    MAXD (float): The maximum distance for the radius.
    directory (str): The directory to save the animation.
    fname (str): The filename to save the animation.
    side_by_side (bool): Whether to display the plots side by side.
    """
    n_agents = len(paths)
    colormap = plt.cm.get_cmap("tab10", n_agents)
    colors = [colormap(i) for i in range(n_agents)]
    background_color = "#ddd9dc"

    # Interpolate the paths.
    step = 0.5
    n_agents = len(paths)
    interpolation = interpolate_paths_with_speeds(paths, speeds, rpositions, step)
    interpolation = np.hstack((interpolation, np.zeros((len(interpolation), 1, 2))))

    # Translate the paths to coordinates.
    coordinates = translate_path_to_coordinates(paths, rpositions)
    collected_rewards, collected_values, collection_idx = get_collection_history(
        interpolation, paths, coordinates, rvalues
    )

    # Calculate the RSSI history.
    rssi_history = calculate_rssi_history(paths, speeds, rpositions, step=step)

    # Create the figure and axes.
    if side_by_side:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))

    fig.patch.set_facecolor(background_color)
    ax1.set_facecolor(background_color)
    ax2.set_facecolor(background_color)
    plt.subplots_adjust(hspace=0.5 if not side_by_side else 0.3, wspace=0.3 if side_by_side else 0.5)

    # Plot rewards on the first axis
    plot_rewards(ax1, rpositions, rvalues)
    ax1.set_title(
        "Paths - score: "
        + str([int(score) for score in scores])
        + f" - {collected_values[0]} collected"
    )
    ax1.grid(True)
    ax1.axis("equal")
    ax1.set_ylim(0, None)
    ax1.set_xlim(0, None)

    # Plot RSSI history on the second axis
    ax2.plot(rssi_history)
    ax2.set_title("RSSI History")
    ax2.grid(True)
    ax2.set_ylim(min(rssi_history) - 1, 0)
    ax2.set_xlim(0, len(rssi_history))

    def update(i):
        """
        Updates the animation.

        Parameters:
        i (int): The current frame index.
        """
        print(i)
        # Find the index of the collection.
        idx = len(collection_idx) - 1
        for j in range(len(collection_idx)):
            if i <= collection_idx[j]:
                idx = j
                break

        # Clear the axes.
        ax1.clear()
        # Plot the rewards.
        plot_rewards(ax1, rpositions, rvalues, collected_rewards[idx])
        # Plot the paths of the agents.
        for k in range(n_agents):
            plot_path(ax1, interpolation[k][i : i + 2], color=colors[k])
            plot_path(ax1, interpolation[k][: i + 1], color=colors[k], show_arrow=False)
        ax1.set_title(
            "Paths - score: "
            + str([int(score) for score in scores])
            + f" - {collected_values[idx]} collected"
        )
        # Plot the maximum distance between the paths.
        plot_max_distance(ax1, interpolation[:, :i + 1])

        ax1.grid(True)
        ax1.axis("equal")
        ax1.set_ylim(0, None)
        ax1.set_xlim(0, None)
        ax1.legend()

        ax2.clear()
        ax2.plot(rssi_history[: i + 1])
        ax2.set_title("RSSI History")
        ax2.grid(True)
        ax2.set_ylim(min(rssi_history) - 1, 0)
        ax2.set_xlim(0, len(rssi_history))

    ani = animation.FuncAnimation(
        fig, update, frames=len(interpolation[0]), repeat=False
    )
    # plt.show()
    ani.save(f"{directory}/{fname}.gif", writer="pillow", fps=22)

    # Save the last frame as a static image
    update(len(interpolation[0]) - 1)
    plt.savefig(f"{directory}/{fname}_last_frame.png")
    plt.close()


def plot_max_distance(ax, individual):
    # Pad the paths to the same size.
    max_size = max(len(ind) for ind in individual)
    individual = [
        np.vstack([ind, np.tile(ind[-1], (max_size - len(ind), 1))])
        for ind in individual
    ]

    # Find the maximum distance between the paths.
    max_distance = 0
    point1 = point2 = None
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            distances = np.linalg.norm(individual[i] - individual[j], axis=1)
            max_dist_index = np.argmax(distances)
            if distances[max_dist_index] > max_distance:
                max_distance = distances[max_dist_index]
                point1 = individual[i][max_dist_index]
                point2 = individual[j][max_dist_index]

    # Plot the maximum distance between the paths.
    if point1 is not None and point2 is not None:
        ax.plot(
            [point1[0], point2[0]], [point1[1], point2[1]], "r--", label="Max Distance"
        )
        ax.scatter(point1[0], point1[1], c="r", marker="o")
        ax.scatter(point2[0], point2[1], c="r", marker="o")


def plot_distances(
    path1, path2, positions, max_distance, num_samples=100, directory=None
):
    """
    Plots the distances between two paths.

    Parameters:
    path1 (list): The first path.
    path2 (list): The second path.
    positions (np.ndarray): The positions of the rewards.
    max_distance (float): The maximum distance.
    num_samples (int): The number of samples.
    directory (str): The directory to save the plot.
    """
    interpolated_paths = interpolate_paths(path1, path2, positions, num_samples)

    distances = np.linalg.norm(interpolated_paths[0] - interpolated_paths[1], axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("grey")
    ax.set_facecolor("grey")
    ax.plot(distances, color="red")
    ax.fill_between(range(len(distances)), distances, color="red", alpha=0.3)
    ax.axhline(
        max_distance,
        color="blue",
        linestyle="--",
        label=f"Max distance: {max_distance}",
    )
    ax.set_ylabel("Distance between agents")
    ax.set_xlabel("Interpolated steps along paths")
    ax.set_title("Interpolated Distances Between Agents")
    plt.grid(True)
    plt.tight_layout()

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/distances.png")


def plot_objectives(logbook):
    """
    Plots the objectives over generations.

    Parameters:
    logbook (deap.tools.Logbook): The logbook containing the data.
    """
    generations = logbook.select("gen")
    max_rewards = logbook.select("max_reward")
    avg_rewards = logbook.select("avg_reward")
    min_distances = logbook.select("min_distance")
    avg_distances = logbook.select("avg_distance")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("grey")
    ax1.set_facecolor("grey")

    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Recompensa", color="tab:blue")
    ax1.plot(
        generations,
        max_rewards,
        label="Máxima Recompensa",
        color="tab:blue",
        linestyle="--",
    )
    ax1.plot(generations, avg_rewards, label="Recompensa Média", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_facecolor("grey")
    ax2.set_ylabel("Distância", color="tab:red")
    ax2.plot(
        generations,
        min_distances,
        label="Mínima Distância",
        color="tab:red",
        linestyle="--",
    )
    ax2.plot(generations, avg_distances, label="Distância Média", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title("Evolução dos Objetivos")
    plt.show()


def plot_pareto_front(archive, directory=None):
    """
    Plots the Pareto front.

    Parameters:
    archive (list): The archive of solutions.
    directory (str): The directory to save the plot.
    """
    scores = [ind.score for ind in archive]
    rewards = [score[0] for score in scores]
    rssi = [score[1] for score in scores]

    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor("grey")
    plt.scatter(rssi, rewards, c="blue", label="Soluções")
    plt.xlabel("RSSI")
    plt.ylabel("Reward")
    plt.title("Pareto Frontier")
    plt.legend()
    plt.grid(True)

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/front.png")

    plt.show()


def plot_pareto_front_evolution(log, directory=None):
    """
    Plots the evolution of the Pareto front.

    Parameters:
    log (list): The log of the Pareto front evolution.
    """
    iterations = len(log)
    fig, ax = plt.subplots()

    # Configuração inicial do gráfico
    scatter = ax.scatter([], [], s=30, alpha=0.6)
    ax.set_xlabel("Maximum RSSI")
    ax.set_ylabel("Percentage of Total Reward Obtained")
    ax.set_title("Pareto Frontier Evolution")
    ax.set_xlim(-100, 0)
    ax.set_ylim(0, 100)

    initial_data = log[0]
    scores_x, scores_y = zip(*[(s[1], s[0]) for s in initial_data["front"]])
    ax.plot(
        scores_x,
        scores_y,
        linewidth=2,
        color="green",
        marker="o",
        markersize=6,
        alpha=0.6,
        label="Pareto Front",
    )

    def update(iteration):
        """
        Updates the animation.

        Parameters:
        iteration (int): The current iteration.
        """
        ax.clear()

        initial_data = log[0]
        scores_x, scores_y = zip(*[(s[1], s[0]) for s in initial_data["front"]])
        ax.plot(
            scores_x,
            scores_y,
            linewidth=2,
            color="green",
            marker="o",
            markersize=6,
            alpha=0.6,
            label="Initial pareto front",
        )

        o = 0
        v = []
        current_data = log[iteration]
        for i in range(len(current_data["front"])):
            for j in range(i + 1, len(current_data["front"])):
                if (
                    current_data["front"][i][0] == current_data["front"][j][0]
                    and current_data["front"][i][1] == current_data["front"][j][1]
                    and i not in v
                    and j not in v
                ):
                    o += 1
                    v.append(j)

        scores_x, scores_y = zip(*[(s[1], s[0]) for s in current_data["front"]])
        ax.plot(
            scores_x,
            scores_y,
            linewidth=2,
            color="purple",
            marker="o",
            markersize=6,
            label="Pareto front",
        )

        if current_data["dominated"]:
            x_dominated, y_dominated = zip(
                *[(s[1], s[0]) for s in current_data["dominated"]]
            )
            ax.scatter(
                x_dominated,
                y_dominated,
                s=30,
                alpha=0.6,
                color="gray",
                label="Dominated",
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Maximum RSSI")
        ax.set_ylabel("Percentage of Total Reward Obtained")
        ax.set_title("Archive Evolution")
        ax.set_title(f"Pareto Frontier Evolution - Iteration {iteration}")
        ax.legend()

    # Cria a animação
    ani = animation.FuncAnimation(
        fig, update, frames=iterations, repeat=False, interval=150
    )

    # Mostra a animação ou salva em um arquivo
    os.makedirs(directory, exist_ok=True)

    ani.save(f"{directory}/pareto_front_evolution.gif", writer="pillow")

    # Save the last frame as a static image
    update(iterations - 1)
    plt.savefig(f"{directory}/pareto_front_evolution_last_frame.png")
    plt.close()


def plot_interpolated_individual(individual: list, maxdist: float, directory=None):
    """
    Plots the interpolated individual paths.

    Parameters:
    individual (list): The paths of the individual.
    maxdist (float): The maximum distance for the radius.
    directory (str): The directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_path(ax, individual[0], color="orange")
    plot_path(ax, individual[1], color="green")

    ax.set_title("Interpolated Individual Paths")
    plt.grid(True)
    plt.axis("equal")
    plt.ylim(0, None)
    plt.xlim(0, None)

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/interpolation.png")


def plot_rssi(path1, path2, positions, num_samples=100, directory=None):
    """
    Plots the RSSI between two paths.

    Parameters:
    path1 (list): The first path.
    path2 (list): The second path.
    positions (np.ndarray): The positions of the rewards.
    num_samples (int): The number of samples.
    directory (str): The directory to save the plot.
    """
    interpolated_paths = interpolate_paths(path1, path2, positions, num_samples)

    distances = calculate_rssi(interpolated_paths[0], interpolated_paths[1], positions)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, color="red")
    ax.fill_between(range(len(distances)), distances, color="red", alpha=0.3)
    ax.set_ylabel("Distance between agents")
    ax.set_xlabel("Interpolated steps along paths")
    ax.set_title("Interpolated RSSI Between Agents")
    plt.grid(True)
    plt.tight_layout()

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/rssi.png")
