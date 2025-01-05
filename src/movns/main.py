import random
import time
import numpy as np
from scipy.spatial.distance import cdist
import plot
from .evaluation import evaluate, update_archive
from .operators import *
from .entities import Solution


def init_solution(num_agents, num_rewards: int) -> tuple:
    val = [np.random.permutation(num_rewards) for _ in range(num_agents)]
    return Solution(val)


def save_stats(front, dominated, log):
    front.sort(key=lambda s: s.score[0])
    dominated.sort(key=lambda s: s.score[0])
    log.append({'front': [s.score for s in front], 'dominated': [s.score for s in dominated]})


def select_solution(archive, front, dominated, visited):
    choosen_set = random.random()
    if choosen_set < 0.5:
        candidates = [s for s in front if id(s) not in visited]
    else:
        candidates = [s for s in dominated if id(s) not in visited]

    if not candidates:
        candidates = [s for s in archive]
        visited.clear()

    solution = random.choice(candidates)
    visited.add(id(solution))
    return solution


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
    archive_max_size = 20
    archive = [solution]
    front = []

    kmax = 4  # Number of neighborhoods

    for k in range(kmax):
        neighbors = local_search(solution, k, rvalues, rpositions, distmx)
        archive[:], front[:], dominated = update_archive(archive, neighbors, archive_max_size)
        save_stats(front, dominated, log)

    visited = set()

    len(archive)

    for it in range(max_it):
        print(it, len(archive))
        # Select neighborhood
        k = it % kmax + 1
        
        # Select solution
        solution = select_solution(archive, front, dominated, visited)

        shaken_solution = perturb_solution(solution, k)
        shaken_solution.score = evaluate(shaken_solution, rvalues, rpositions, distmx)

        neighbors1 = local_search(shaken_solution, k, rvalues, rpositions, distmx)

        solution1, solution2 = random.sample(front, 2)
        neighbors2 = solution_relinking(
            solution1, solution2, rvalues, rpositions, distmx
        )

        archive[:], front[:], dominated = update_archive(archive, neighbors1 + neighbors2, archive_max_size)

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
    max_it: int = 200,
    num_agents: int = 2,
    seed: int = 42,
):
    Solution.set_parameters(begin, end, budget)

    # Matrix of distances between rewards
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    archive, front, log = movns(
        num_agents, num_rewards, rvalues, rpositions, distmx, max_it, seed
    )
    archive.sort(key=lambda solution: solution.score[0])

    directory = 'imgs/movns/movns'
    plot.plot_pareto_front(archive, directory)
    plot.plot_parento_front_evolution(log)

    bounded_paths = [s.get_solution_paths(distmx) for s in front]
    print(archive[0].score, archive[len(archive) // 2].score, archive[len(archive) - 1].score)

    for i, bounded_path in enumerate(bounded_paths):
        plot.plot_paths_with_rewards(rpositions, rvalues, [bounded_path[0], bounded_path[1]], 4, directory=directory, fname=f'paths{i}')

    return bounded_paths
