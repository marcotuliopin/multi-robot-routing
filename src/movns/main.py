import random
import numpy as np
from scipy.spatial.distance import cdist
import plot
from .evaluation import evaluate, update_archive
from .operators import *
from .entities import Solution


def init_solution(num_agents, num_rewards: int) -> tuple:
    val = [np.random.permutation(num_rewards) for _ in range(num_agents)]
    return Solution(val)


def movns(
    num_agents: int,
    num_rewards: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    max_it: int,
    seed: int,
):
    random.seed(seed)

    # Each solution is composed of num_agents paths
    solution: Solution = init_solution(num_agents, num_rewards)
    solution.score = evaluate(solution, rvalues, rpositions, distmx)

    # The archive is composed of the non-dominated solutions
    archive_max_size = 30
    archive = [solution]

    kmax = 4  # Number of neighborhoods

    for k in range(kmax):
        neighbors = local_search(solution, k, rvalues, rpositions, distmx)
        archive[:] = update_archive(archive, neighbors, archive_max_size)

    visited = set()

    len(archive)

    for it in range(max_it):
        print(it, len(archive))
        k = it % kmax + 1

        candidates = [s for s in archive if id(s) not in visited]
        if not candidates:
            candidates = [s for s in archive]
            visited.clear()

        solution = random.choice(candidates)
        visited.add(id(solution))

        shaken_solution = perturb_solution(solution, k)
        shaken_solution.score = evaluate(shaken_solution, rvalues, rpositions, distmx)
        neighbors1 = local_search(shaken_solution, k, rvalues, rpositions, distmx)

        solution1, solution2 = random.sample(archive, 2)
        neighbors2 = solution_relinking(
            solution1, solution2, rvalues, rpositions, distmx
        )
        archive[:] = update_archive(archive, neighbors1 + neighbors2, archive_max_size)

    return archive


def main(
    num_rewards: int,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    budget: int,
    begin: int = 0,
    end: int = 0,
    max_it: int = 300,
    num_agents: int = 2,
    seed: int = 42,
):
    Solution.set_parameters(begin, end, budget)

    # Matrix of distances between rewards
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    archive: list[Solution] = movns(
        num_agents, num_rewards, rvalues, rpositions, distmx, max_it, seed
    )
    archive.sort(key=lambda solution: solution.score[0])

    plot.plot_pareto_front(archive)

    bounded_paths = [s.get_solution_paths(distmx) for s in archive]
    print(archive[0].score, archive[len(archive) // 2].score, archive[len(archive) - 1].score)

    return bounded_paths
