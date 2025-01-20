import random
import time
import plot
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from .evaluation import evaluate, update_archive
from .operators import *
from .entities import Solution
import pickle

# Time complexity: O(n * m), where n is the number of agents and m is the number of rewards.
# Space complexity: O(n * m), as it stores the initial solution.
def init_solution(num_agents, num_rewards: int) -> tuple:
    val = [np.random.permutation(num_rewards) for _ in range(num_agents)]
    return Solution(val)

# Time complexity: O(n log n), where n is the number of solutions in the front and dominated lists.
# Space complexity: O(n), as it stores the sorted solutions.
def save_stats(front, dominated, log):
    front.sort(key=lambda s: s.score[0])
    dominated.sort(key=lambda s: s.score[0])
    log.append({'front': [s.score for s in front], 'dominated': [s.score for s in dominated]})

# Time complexity: O(n), where n is the number of solutions in the front and dominated lists.
# Space complexity: O(1), as it uses a constant amount of additional space.
def select_solution(front, dominated):
    choosen_set = random.random()
    if choosen_set < 0.9 or not dominated:
        candidates = get_candidates(front)
    else:
        candidates = get_candidates(dominated)

    solution = random.choice(candidates)
    solution.visited = True

    return solution.copy()

# Time complexity: O(n), where n is the number of solutions.
# Space complexity: O(n), as it stores the candidates.
def get_candidates(solutions):
    candidates = [s for s in solutions if not s.visited]
    if not candidates:
        for s in solutions:
            s.visited = False
        candidates = solutions
    return candidates

# Time complexity: O(n * m^2 * k), where n is the number of agents, m is the number of rewards, and k is the number of iterations.
# Space complexity: O(n * m), as it stores the solutions and the archive.
def movns(
    num_agents: int,
    num_rewards: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    max_it: int,
    seed: int,
):
    start = time.time()

    random.seed(seed)
    log = []

    # Each solution is composed of num_agents paths
    solution: Solution = init_solution(num_agents, num_rewards)
    solution.score = evaluate(solution, rvalues, rpositions, distmx)

    # The archive is composed of the non-dominated solutions
    archive_max_size = 25
    archive = [solution]
    front = []

    kmax = 4  # Number of neighborhoods

    print(f"Initializing neighborhood")
    for k in tqdm(range(kmax), desc="Progress", unit="neighborhood"):
        neighbors = local_search(solution, k, rvalues, rpositions, distmx)
        archive, front, dominated = update_archive(archive, neighbors, archive_max_size)
    save_stats(front, dominated, log)

    for it in tqdm(range(max_it), desc="Progress", unit="iteration"):
        start_it = time.perf_counter()

        # Select neighborhood
        k = np.random.randint(1, kmax)
        
        solution = select_solution(front, dominated) # O(n) where n is the number of solutions in the front and dominated lists

        # First phase
        shaken_solution = perturb_solution(solution, k) # O(n * m) where n is the number of paths and m is the number of points in each path
        shaken_solution.score = evaluate(shaken_solution, rvalues, rpositions, distmx) # O(n * m) where n is the number of paths and m is the number of points in each path
        neighbors1 = local_search(shaken_solution, k, rvalues, rpositions, distmx) # O(n * m^2) where n is the number of paths and m is the number of points in each path

        # Second phase
        neighbors2 = []
        if len(front) > 1:
            solution1, solution2 = random.sample(front, 2)
            neighbors2 = solution_relinking(
                solution1.copy(), solution2.copy(), rvalues, rpositions, distmx
            ) # O(n * m^2) where n is the number of paths and m is the number of points in each path

        archive, front, dominated = update_archive(archive, neighbors1 + neighbors2, archive_max_size) # O(n log n) where n is the number of solutions in the archive and neighbors
        end_it = time.perf_counter()
        with open(f"tests/it_time_{num_agents}_bdg_{Solution._BUDGET}.txt", "a") as f:
            f.write(f"{str(end_it - start_it)}\n")

        save_stats(front, dominated, log)

    end = time.time() - start
    print(f"Execution time: {end / 60:.2f} minutes")

    return archive, front, log

def main(
    num_rewards: int,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    budget: int,
    begin: int = 0,
    end: int = 0,
    max_it: int = 100,
    num_agents: int = 4,
    seed: int = 42,
):
    Solution.set_parameters(begin, end, budget)

    total_rewards = rvalues.sum()
    percentual_values = (rvalues / total_rewards) * 100

    # Matrix of distances between rewards
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    archive, front, log = movns(
        num_agents, num_rewards, percentual_values, rpositions, distmx, max_it, seed
    )
    archive.sort(key=lambda solution: solution.score[0])


    bounded_paths = [s.get_solution_paths(distmx) for s in front]
    scores = [s.score for s in front]

    for i, bounded_path in enumerate(bounded_paths):
        with open(f'out/bounded_path_{i}_{num_agents}_agents_{budget}_bgt.pkl', 'wb') as f:
            pickle.dump(bounded_path, f)
            pickle.dump(scores[i], f)

    # directory = 'imgs/movns/movns'

    # plot.plot_pareto_front_evolution(log)

    # for i, bounded_path in enumerate(bounded_paths):
    #     print('Bounded Path', i)
    #     plot.plot_paths_with_rewards(rpositions, rvalues, bounded_path, scores[i], 4, directory=directory, fname=f'{num_agents}_agents_{budget}_bgt_paths{i}')

    return bounded_paths
