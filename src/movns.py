import os
import random
import time
import plot
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
from .evaluation import evaluate, update_archive
from .operators import *
from .entities import Solution, Neighborhood

archive_max_size = 40

def save_stats(front, dominated, log):
    front.sort(key=lambda s: (s.score[0], s.score[1], s.score[2]))
    dominated.sort(key=lambda s: (s.score[0], s.score[1], s.score[2]))
    log.append(
        {"front": [s.score for s in front], "dominated": [s.score for s in dominated]}
    )


def select_solution(front, dominated):
    choosen_set = np.random.random()
    if choosen_set < 0.9 or not dominated:
        candidates = get_candidates(front)
    else:
        candidates = get_candidates(dominated)

    solution = random.choice(candidates)
    solution.visited = True

    return solution.copy()


def get_candidates(solutions):
    candidates = [s for s in solutions if not s.visited]
    if not candidates:
        for s in solutions:
            s.visited = False
        candidates = solutions
    return candidates


def movns(
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    total_time: int,
    seed: int,
):
    np.random.seed(seed)

    neighborhood = Neighborhood()

    solution = Solution(distmx=distmx, rvalues=rvalues)
    solution.score = evaluate(solution, rvalues, rpositions, distmx)

    archive = [solution]
    front = []
    log = []

    # Initialize the archive by exploring the neighborhood of the initial solution.
    print("Initializing archive...")
    for neighborhood_id in range(neighborhood.num_neighborhoods):
        neighbors = local_search(
            solution, neighborhood, neighborhood_id, rvalues, rpositions, distmx
        )
        archive, front, dominated = update_archive(archive, neighbors, archive_max_size)

    save_stats(front, dominated, log)

    # Main loop.
    print("Running main loop...")
    progress_bar = tqdm(total=total_time, desc="Progress", unit="sec", dynamic_ncols=True)

    init_time = time.perf_counter()
    while time.perf_counter() - init_time < total_time:
        solution = select_solution(front, dominated)

        for neighborhood_id in range(neighborhood.num_neighborhoods):
            elapsed_time = time.perf_counter() - init_time
            progress_bar.n = int(elapsed_time)
            progress_bar.refresh()
            if elapsed_time > total_time:
                break

            shaken_solution = perturb_solution(solution, neighborhood, rvalues, rpositions, distmx)
            neighbors = local_search(shaken_solution, neighborhood, neighborhood_id, rvalues, rpositions, distmx)

            archive, front, dominated = update_archive(archive, [shaken_solution] + neighbors, archive_max_size)

            save_stats(front, dominated, log)

    return archive, front, log


def main(
    rpositions: np.ndarray, # Rewards coordinates
    rvalues: np.ndarray, # Rewards values
    budget: list[int],
    begin: int = -1,
    end: int = -1,
    total_time: int = 300,
    num_agents: int = 1,
    speeds: list = [1],
    seed: int = 42,
):
    Solution.set_parameters(begin, end, num_agents, budget, speeds)

    # Matrix of distances between rewards
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    # Run the algorithm
    archive, front, log = movns(rvalues, rpositions, distmx, total_time, seed)
    archive.sort(key=lambda solution: solution.score[0])

    paths = [s.get_solution_paths() for s in front]
    scores = [s.score for s in front]

    # Store the results of the front for further analysis.
    for i, path in enumerate(paths):
        out_dir = f"out/front/{num_agents}_agents/{max(budget)}_bgt"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/scores.pkl", "ab") as f:
            pickle.dump(scores[i], f)
        with open(f"{out_dir}/paths.pkl", "ab") as f:
            pickle.dump(path, f)

    directory = "imgs/movns/"

    # Create an animation of the Pareto front evolution.
    print("Plotting Front Evolution Animation...")
    plot.plot_pareto_front_evolution_3d(log, directory+f"/animations/{num_agents}_agents/{max(budget)}_bgt")
    plot.plot_pareto_front_evolution_2d(log, directory+f"/animations/{num_agents}_agents/{max(budget)}_bgt")

    # Plot each path of the Pareto front.
    print("Plotting Paths...")
    for i, path in enumerate(paths):
        plot.plot_paths_with_rewards(
            rpositions,
            rvalues,
            path,
            scores[i],
            directory=directory+f"/paths/{num_agents}_agents/{max(budget)}_bgt",
            fname=f"{i}",
        )

    return paths
