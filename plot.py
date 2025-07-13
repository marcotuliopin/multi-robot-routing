import os
from matplotlib import animation, pyplot as plt
import matplotlib.gridspec as gridspec
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
    rewards_1 = [i for i in range(len(reward_p) - 2) if i not in not_plot]
    scatter = ax.scatter(
        reward_p[rewards_1, 0],
        reward_p[rewards_1, 1],
        c=reward_value[rewards_1],
        cmap='viridis',
        edgecolors="black",
        s=700,
        label="PoI",
        marker="D",
        vmin=1,
        vmax=10
    )
    ax.scatter(
        reward_p[-1, 0],
        reward_p[-1, 1],
        c="#ff6361",
        edgecolors="black",
        s=800,
        label="Start",
        marker="s",
    )
    ax.scatter(
        reward_p[-2, 0],
        reward_p[-2, 1],
        c="#58508d",
        edgecolors="black",
        s=800,
        label="Goal",
        marker="X",
    )
    rewards_2 = [i for i in range(len(reward_p)) if i in not_plot]
    ax.scatter(
        reward_p[rewards_2, 0],
        reward_p[rewards_2, 1],
        c="#58508d",
        edgecolors="black",
        label="Visited PoI",
        s=700,
        marker="o",
    )
    return scatter
    # Adicionar a colorbar


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
            ax.plot([prev[0], curr[0]], [prev[1], curr[1]], linewidth=5, color=color)

        dx, dy = curr[0] - prev[0], curr[1] - prev[1]

        norm = np.sqrt(dx**2 + dy**2)
        dx_fixed = (dx / norm) * .5
        dy_fixed = (dy / norm) * .5

        if show_arrow:
            ax.quiver(
                prev[0],
                prev[1],
                dx_fixed,
                dy_fixed,
                angles="xy",
                scale_units="xy",
                scale=1,  # Keep scale=1 for direct control
                color=color,
                width=0.01,  # Increase arrow thickness
                headlength=20,  # Bigger arrowhead
                headwidth=20,
                headaxislength=19,
            )
        prev = curr
    plot_max_distance(ax, [path])


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

    fig, ax = plt.subplots(figsize=(20, 20))
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
        + str([float(score) for score in scores])
        + "- length: "
        + str([float(l) for l in length])
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
                abs(coordinates[j][idx][z] - interpolation[j][i][z]) < 0.1
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
    update_rewards=False,
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
    colors = ["#003f5c", "#ff6361", "#ffa600"]
    background_color = "#ffffff"

    # Interpolate the paths.
    step = 20
    n_agents = len(paths)
    interpolation = interpolate_paths_with_speeds(paths, speeds, rpositions, step)

    # Translate the paths to coordinates.
    coordinates = translate_path_to_coordinates(paths, rpositions)
    collected_rewards, collected_values, collection_idx = get_collection_history(
        interpolation, paths, coordinates, rvalues
    )

    if not update_rewards:
        collected_rewards = []

    # Calculate the RSSI history.
    rssi_history = calculate_rssi_history(paths, speeds, rpositions, step=step)

    x1_lim, y1_lim = max(rpositions[:, 0]) + 1, max(rpositions[:, 1]) + 1

    # Create the figure and axes.
    fig = plt.figure(figsize=(24, 11))  # Define o tamanho da figura
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.5, 1])  

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax_legend = fig.add_subplot(gs[0, 1])

    fig.patch.set_facecolor(background_color)
    ax1.set_facecolor("#ffcccc")
    ax2.set_facecolor(background_color)
    ax_legend.set_facecolor(background_color)
    plt.subplots_adjust(hspace=0.6, wspace=0.6)

    # Plot rewards on the first axis
    scatter = plot_rewards(ax1, rpositions, rvalues)
    ax1.set_title(
        f"R: {(int(scores[0]))} | "
        + f"MIN RSSI: {int(scores[1])} | "
        + f"E: {int(scores[2])} "
        , fontsize=30
    )
    ax1.grid(True)
    ax1.set_ylim(0, y1_lim)
    ax1.set_xlim(0, x1_lim)
    ax1.tick_params(axis="both", labelsize=20)

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Reward Value', fontsize=40)
    cbar.ax.tick_params(labelsize=40)

    # Plot RSSI history on the second axis
    ax2.plot(rssi_history)
    # ax2.grid(True)
    ax2.set_ylim(min(rssi_history) - 1, 0)
    ax2.set_xlim(0, len(rssi_history))
    # ax2.set_xlabel("Frame", fontsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    plt.tight_layout()

    ax_legend.axis("off")
    handles, labels = ax1.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", fontsize=50)

    def update(i):
        """
        Updates the animation.

        Parameters:
        i (int): The current frame index.
        """
        # Find the index of the collection.
        idx = len(collection_idx) - 1
        for j in range(len(collection_idx)):
            if i <= collection_idx[j]:
                idx = j
                break

        # Clear the axes.
        ax1.clear()
        ax1.grid(True)
        ax1.set_ylim(0, y1_lim)
        ax1.set_xlim(0, x1_lim)
        ax1.tick_params(axis="both", labelsize=20)

        # Plot the rewards.
        _ = plot_rewards(ax1, rpositions, rvalues, collected_rewards[idx])
        # Plot the paths of the agents.
        for k in range(n_agents):
            plot_path(ax1, interpolation[k][: i + 1], color=colors[k], show_arrow=False)
            plot_path(ax1, interpolation[k][i : i + 2], color=colors[k], show_arrow=True)
        ax1.set_title(
            f"R: {(int(scores[0]))} | "
            + f"MIN RSSI: {int(scores[1])} | "
            + f"E: {int(scores[2])} "
            , fontsize=50
        )
        # Plot the maximum distance between the paths.
        plot_max_distance(ax1, interpolation[:, :i + 1])
        # ax1.legend(loc='lower right', fontsize=15)

        ax2.clear()
        ax2.plot(rssi_history[: i + 1], color="red", alpha=0.7, linewidth=5)
        # ax2.grid(True)
        ax2.set_ylim(min(rssi_history) - 1, 0)
        ax2.set_xlim(0, len(rssi_history))
        # ax2.set_xlabel("Frame", fontsize=20)
        ax2.set_title(f"RSSI: {rssi_history[i]: .2f}", fontsize=50)
        ax2.tick_params(axis="y", labelsize=40)
        plt.tight_layout()
        ax_legend.legend(handles, labels, loc="center", fontsize=50)

        if i % 10 == 0:
            plt.savefig(f"{directory}/{fname}_frame_{i}.png")

    ani = animation.FuncAnimation(
        fig, update, frames=len(interpolation[0]), repeat=False
    )
    # plt.show()
    ani.save(f"{directory}/{fname}.mp4", writer="ffmpeg", fps=20)

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
            [point1[0], point2[0]], [point1[1], point2[1]], "r--", label="Max Distance", linewidth=3
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


def plot_pareto_front_evolution_3d(log, directory=None):
    """
    Plots the evolution of the Pareto front with three different scores.

    Parameters:
    log (list): The log of the Pareto front evolution.
    """
    iterations = len(log)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Configuração inicial do gráfico
    scatter = ax.scatter([], [], [], s=30, alpha=0.6)
    ax.set_xlabel("Percentage of Collected Rewards")
    ax.set_ylabel("Distance between agents")
    ax.set_zlabel("Path Length")
    ax.set_title("Pareto Frontier Evolution")
    ax.set_xlim(0, 100)
    ax.set_ylim(-100, 0)
    ax.set_zlim(-200, 0)

    initial_data = log[0]
    scores_x, scores_y, scores_z = zip(*[(s[0], s[1], s[2]) for s in initial_data["front"]])
    ax.plot(
        scores_x,
        scores_y,
        scores_z,
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
        scores_x, scores_y, scores_z = zip(*[(s[0], s[1], s[2]) for s in initial_data["front"]])
        ax.plot(
            scores_x,
            scores_y,
            scores_z,
            linewidth=2,
            color="green",
            marker="o",
            markersize=6,
            alpha=0.6,
            label="Initial pareto front",
        )

        current_data = log[iteration]
        scores_x, scores_y, scores_z = zip(*[(s[0], s[1], s[2]) for s in current_data["front"]])
        ax.plot(
            scores_x,
            scores_y,
            scores_z,
            linewidth=2,
            color="purple",
            marker="o",
            markersize=6,
            label="Pareto front",
        )

        if current_data["dominated"]:
            x_dominated, y_dominated, z_dominated = zip(
                *[(s[0], s[1], s[2]) for s in current_data["dominated"]]
            )
            ax.scatter(
                x_dominated,
                y_dominated,
                z_dominated,
                s=30,
                alpha=0.6,
                color="gray",
                label="Dominated",
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(-100, 0)
        ax.set_zlim(-200, 0)
        ax.set_xlabel("Percentage of Collected Rewards")
        ax.set_ylabel("Distance between agents")
        ax.set_zlabel("Path Length")
        ax.set_title(f"Pareto Frontier Evolution - Iteration {iteration}")
        ax.legend()

    # Cria a animação
    ani = animation.FuncAnimation(
        fig, update, frames=iterations, repeat=False, interval=150
    )

    # Mostra a animação ou salva em um arquivo
    os.makedirs(directory, exist_ok=True)

    ani.save(f"{directory}/pareto_front_evolution_3d.gif", writer="pillow")

    # Save the last frame as a static image
    update(iterations - 1)
    plt.savefig(f"{directory}/pareto_front_evolution_3d_last_frame.png")
    plt.close()


def plot_pareto_front_evolution_2d(log, directory=None):
    """
    Plots the evolution of the Pareto front with two different scores.

    Parameters:
    log (list): The log of the Pareto front evolution.
    """
    iterations = len(log)
    fig, ax = plt.subplots()

    # Configuração inicial do gráfico
    scatter = ax.scatter([], [], s=30, alpha=0.6)
    ax.set_xlabel("Distance between agents")
    ax.set_ylabel("Percentage of Collected Rewards")
    ax.set_title("Pareto Frontier Evolution")
    ax.set_xlim(-100, 0)
    ax.set_ylim(0, 100)

    initial_data = log[0]
    scores_y, scores_x = zip(*[(s[0], s[1]) for s in initial_data["front"]])
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
        scores_y, scores_x = zip(*[(s[0], s[1]) for s in initial_data["front"]])
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

        current_data = log[iteration]
        scores_y, scores_x = zip(*[(s[0], s[1]) for s in current_data["front"]])
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
            y_dominated, x_dominated = zip(
                *[(s[0], s[1]) for s in current_data["dominated"]]
            )
            ax.scatter(
                x_dominated,
                y_dominated,
                s=30,
                alpha=0.6,
                color="gray",
                label="Dominated",
            )

        ax.set_xlim(-100, 0)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage of Collected Rewards")
        ax.set_xlabel("Distance between agents")
        ax.set_title(f"Pareto Frontier Evolution - Iteration {iteration}")
        ax.legend()

    # Cria a animação
    ani = animation.FuncAnimation(
        fig, update, frames=iterations, repeat=False, interval=150
    )

    # Mostra a animação ou salva em um arquivo
    os.makedirs(directory, exist_ok=True)

    ani.save(f"{directory}/pareto_front_evolution_2d.gif", writer="pillow")

    # Save the last frame as a static image
    update(iterations - 1)
    plt.savefig(f"{directory}/pareto_front_evolution_2d_last_frame.png")
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
