import os
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

def save_stats(log, front):
    # print(int(max(s.score[0] for s in front)), int(max(s.score[1] for s in front)))
    log.append([max(s.score[0] for s in front), max(s.score[1] for s in front)])


def select_solution(front, dominated, it):
    choosen_set = np.random.random()
    if choosen_set < 0.9 or not dominated:
        candidates = get_candidates(front)
    else:
        candidates = get_candidates(dominated)
    if it % 2 == 0: # Get the solution with the highest reward.
        probabilities = np.array([s.score[0] for s in candidates])
        probabilities = probabilities / np.sum(probabilities)
        solution = np.random.choice(candidates, p=probabilities)
    else: # Get the solution with the highest RSSI
        probabilities = np.array([1 / s.score[1] for s in candidates])
        probabilities = probabilities / np.sum(probabilities)
        solution = np.random.choice(candidates)
    solution.visited = True

    return solution


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
    max_it: int = 100,
    algorithm: str = "unique_vis",
):
    np.random.seed(seed)

    neighborhood = Neighborhood(algorithm)

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

    save_stats(log, front)

    # Main loop.
    print("Running main loop...")
    init_time = time.perf_counter()

    progress_bar_it = tqdm(total=max_it, desc="Progress", unit="it", dynamic_ncols=True)
    it = 0

    while it < max_it and time.perf_counter() - init_time < total_time:
        solution = select_solution(front, dominated, it)
        
        for neighborhood_id in range(neighborhood.num_neighborhoods):
            elapsed_time = time.perf_counter() - init_time
            if elapsed_time > total_time:
                break

            shaken_solution = perturb_solution(solution, neighborhood, rvalues, rpositions, distmx)
            neighbors = local_search(shaken_solution, neighborhood, neighborhood_id, rvalues, rpositions, distmx)

            archive, front, dominated = update_archive(archive, [shaken_solution] + neighbors, archive_max_size)

            save_stats(log, front)
        it += 1
        progress_bar_it.update(1)

    print(f"Finished running in {time.perf_counter() - init_time:.2f} seconds.")

    return archive, front, log


def main(
    rpositions: np.ndarray, # Rewards coordinates
    rvalues: np.ndarray, # Rewards values
    budget: list[int],
    map: str,
    out: str,
    begin: int = -1,
    end: int = -2,
    total_time: int = 600,
    num_agents: int = 1,
    speeds: list = [1],
    seed: int = 42,
    max_it: int = 100,
    algorithm: str = "unique_vis",
):
    Solution.set_parameters(begin, end, num_agents, budget, speeds)

    # Matrix of distances between rewards
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    # Run the algorithm
    archive, front, log = movns(rvalues, rpositions, distmx, total_time, seed, max_it, algorithm)
    archive.sort(key=lambda solution: solution.score[0])

    paths = [s.get_solution_paths() for s in front]
    scores = np.array([s.score for s in front])
    # scores[:, 0] = scores[:, 1] / np.sum(rvalues) * 100

    # Store the results of the front for further analysis.
    for i, path in enumerate(paths):
        out_dir = f"{out}/{map}/{max(speeds)}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/scores.pkl", "ab") as f:
            pickle.dump(scores[i], f)
        with open(f"{out_dir}/paths.pkl", "ab") as f:
            pickle.dump(path, f)
    with open(f"{out_dir}/log.pkl", "ab") as f:
        pickle.dump(log, f)
    
    print(max(scores[:, 0]))
    
    directory = "imgs/"

    # # Create an animation of the Pareto front evolution.
    # print("Plotting Front Evolution Animation...")
    # plot.plot_pareto_front_evolution_3d(log, directory+f"/animations/{num_agents}_agents/{max(budget)}_bgt")
    # plot.plot_pareto_front_evolution_2d(log, directory+f"/animations/{num_agents}_agents/{max(budget)}_bgt")

    # Plot each path of the Pareto front.
    print("Plotting Paths...")
    paths_dir = f"paths/{map}/"
    for i, path in enumerate(paths):
        path_dir = directory + paths_dir
        os.makedirs(path_dir, exist_ok=True)
        num_files = len([name for name in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, name))])
        plot.plot_paths_with_rewards(
            rpositions,
            rvalues,
            path,
            scores[i],
            directory=directory+paths_dir,
            fname=f"{num_files + 1}",
        )

    return paths
