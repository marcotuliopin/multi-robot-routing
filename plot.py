import os
from matplotlib import animation, pyplot as plt
from utils import interpolate_paths, translate_path_to_coordinates, calculate_rssi
import numpy as np


def plot_rewards(ax, reward_p: np.ndarray, reward_value: np.ndarray, not_plot: set = set()):
    rewards_to_plot = [i for i in range(len(reward_p)) if i not in not_plot]
    ax.scatter(reward_p[rewards_to_plot, 0], reward_p[rewards_to_plot, 1], c="b", s=20, label="Rewards")
    for i, (x, y) in enumerate(reward_p):
        if i in not_plot:
            continue
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
    show_arrow=True,
):
    prev = path[0]

    for curr in path[1:]:
        if show_path:
            ax.plot([prev[0], curr[0]], [prev[1], curr[1]], linewidth=2, color=color)
            
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            if show_arrow:
                ax.quiver(
                    prev[0], prev[1], dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    color=color, width=0.005, headwidth=3, headlength=5
                )

        if show_radius:
            circle = plt.Circle((curr[0], curr[1]), maxdist, color=color, alpha=0.04)
            ax.add_patch(circle)

        prev = curr


def get_path_length(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def plot_paths_with_rewards(rpositions, rvalues, individual, scores, MAXD, directory=None, fname=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_rewards(ax, rpositions, rvalues)
    
    individual = translate_path_to_coordinates(individual, rpositions)
    length = [get_path_length(ind) for ind in individual]
    plot_path(ax, individual[0], MAXD, color='orange')
    plot_path(ax, individual[1], MAXD, color='green')

    ax.set_title("Individual Paths - score: " + str([int(score) for score in scores]) + "- length: " + str([int(l) for l in length]))
    plt.grid(True)
    plt.axis('equal')
    plt.ylim(0, None)
    plt.xlim(0, None)

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/{fname if fname else 'paths'}.png')
    plt.close()


def plot_animated_paths(rpositions, rvalues, individual, scores, MAXD, directory=None, fname=None):
    interpolation = interpolate_paths(individual[0], individual[1], rpositions, 1)
    interpolation = np.hstack((interpolation, np.zeros((len(interpolation), 1, 2))))

    coordinates = translate_path_to_coordinates(individual, rpositions)
    length = [get_path_length(i) for i in coordinates]

    collected_rewards = set()
    collected_value = 0
    path_idx = [0, 0]

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_rewards(ax, rpositions, rvalues)
    
    ax.set_title("Individual Paths - score: " + str([int(score) for score in scores]) + f" - {collected_value} collected")
    plt.grid(True)
    plt.axis('equal')
    plt.ylim(0, None)
    plt.xlim(0, None)

 
    def update(i):
        nonlocal collected_value

        for j, idx in enumerate(path_idx):
            if idx >= len(individual[j]):
                continue
            if all(abs(coordinates[j][idx][z] - interpolation[j][i][z]) < 1 for z in range(2)):
                collected_rewards.add(individual[j][idx])
                path_idx[j] += 1
                collected_value += rvalues[individual[j][idx]]

        ax.clear()

        ax.set_title("Individual Paths - score: " + str([int(score) for score in scores]) + f" - {collected_value} collected")

        plt.grid(True)
        plt.axis('equal')
        plt.ylim(0, None)
        plt.xlim(0, None)

        plot_rewards(ax, rpositions, rvalues, collected_rewards)
        plot_path(ax, interpolation[0][i:i+2], MAXD, color='orange')
        plot_path(ax, interpolation[1][i:i+2], MAXD, color='green')
        plot_path(ax, interpolation[0][:i+1], MAXD, color='orange', show_arrow=False)
        plot_path(ax, interpolation[1][:i+1], MAXD, color='green', show_arrow=False)

    ani = animation.FuncAnimation(
        fig, update, frames=len(interpolation[0]), repeat=False, interval=150
    )

    # Mostra a animação ou salva em um arquivo
    plt.show()
    os.makedirs(directory, exist_ok=True)
    ani.save(f"{directory}/{fname}.gif", writer="pillow")
    


def plot_distances(path1, path2, positions, max_distance, num_samples=100, directory=None):
    interpolated_paths = interpolate_paths(path1, path2, positions, num_samples)
    
    distances = np.linalg.norm(interpolated_paths[0] - interpolated_paths[1], axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, color='red')
    ax.fill_between(range(len(distances)), distances, color='red', alpha=0.3)
    ax.axhline(max_distance, color="blue", linestyle="--", label=f"Max distance: {max_distance}")
    ax.set_ylabel("Distance between agents")
    ax.set_xlabel("Interpolated steps along paths")
    ax.set_title("Interpolated Distances Between Agents")
    plt.grid(True)
    plt.tight_layout()

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/distances.png')


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


def plot_pareto_front(archive, directory=None):
    scores = [ind.score for ind in archive]
    rewards = [score[0] for score in scores]
    rssi = [score[1] for score in scores]

    plt.figure(figsize=(8, 6))
    plt.scatter(rssi, rewards, c="blue", label="Soluções")
    plt.xlabel("RSSI")
    plt.ylabel("Reward")
    plt.title("Pareto Frontier")
    plt.legend()
    plt.grid(True)

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}/front.png')

    plt.show()


def plot_pareto_front_evolution(log):
    iterations = len(log)
    fig, ax = plt.subplots()

    # Configuração inicial do gráfico
    scatter = ax.scatter([], [], s=30, alpha=0.6)
    ax.set_xlabel('Maximum RSSI')
    ax.set_ylabel('Percentage of Total Reward Obtained')
    ax.set_title('Pareto Frontier Evolution')
    ax.set_xlim(-60, 0)
    ax.set_ylim(0, 100)

    initial_data = log[0]
    scores_x, scores_y = zip(*[(s[1], s[0]) for s in initial_data['front']])
    ax.plot(scores_x, scores_y, linewidth=2, color='green', marker='o', markersize=6, alpha=0.6, label='Pareto Front')


    def update(iteration):
        ax.clear()

        initial_data = log[0]
        scores_x, scores_y = zip(*[(s[1], s[0]) for s in initial_data['front']])
        ax.plot(scores_x, scores_y, linewidth=2, color='green', marker='o', markersize=6, alpha=0.6, label='Initial pareto front')

        o = 0
        v = []
        current_data = log[iteration]
        for i in range(len(current_data['front'])):
            for j in range(i + 1, len(current_data['front'])):
                if current_data['front'][i][0] == current_data['front'][j][0] and current_data['front'][i][1] == current_data['front'][j][1] and i not in v and j not in v:
                    o += 1
                    v.append(j)
                
        scores_x, scores_y = zip(*[(s[1], s[0]) for s in current_data['front']])
        ax.plot(scores_x, scores_y, linewidth=2, color='purple', marker='o', markersize=6, label='Pareto front')

        if current_data['dominated']:
            x_dominated, y_dominated = zip(*[(s[1], s[0]) for s in current_data['dominated']])
            ax.scatter(x_dominated, y_dominated, s=30, alpha=0.6, color='gray', label='Dominated')

        ax.set_xlim(-60, 0)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Maximum RSSI')
        ax.set_ylabel('Percentage of Total Reward Obtained')
        ax.set_title('Archive Evolution')
        ax.set_title(f'Pareto Frontier Evolution - Iteration {iteration}')
        ax.legend()

    # Cria a animação
    ani = animation.FuncAnimation(
        fig, update, frames=iterations, repeat=False, interval=150
    )

    # Mostra a animação ou salva em um arquivo
    plt.show()
    os.makedirs("imgs/movns/movns/", exist_ok=True)

    ani.save("imgs/movns/movns/animacao.gif", writer="pillow")


def plot_interpolated_individual(
    individual: list,
    maxdist: float,
    directory=None
):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_path(ax, individual[0], maxdist, color='orange')
    plot_path(ax, individual[1], maxdist, color='green')

    ax.set_title("Interpolated Individual Paths")
    plt.grid(True)
    plt.axis('equal')
    plt.ylim(0, None)
    plt.xlim(0, None)

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/interpolation.png')


def plot_rssi(path1, path2, positions, num_samples=100, directory=None):
    interpolated_paths = interpolate_paths(path1, path2, positions, num_samples)
    
    distances = calculate_rssi(interpolated_paths[0], interpolated_paths[1], positions)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, color='red')
    ax.fill_between(range(len(distances)), distances, color='red', alpha=0.3)
    ax.set_ylabel("Distance between agents")
    ax.set_xlabel("Interpolated steps along paths")
    ax.set_title("Interpolated RSSI Between Agents")
    plt.grid(True)
    plt.tight_layout()

    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/rssi.png')