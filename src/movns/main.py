import random
import numpy as np
from scipy.spatial.distance import cdist
from src.shared.utils import get_last_valid_idx, get_path_length
from src.movns.evaluation import evaluate
from src.movns.operators import init_solution, local_search, perturb_solution
from src.movns.entity.Solution import Solution

def movns(
    num_agents: int,
    num_rewards: int,
    rvalues: np.ndarray,
    budget: int,
    distmx: np.ndarray,
    max_no_improve: int,
    seed: int,
):
    random.seed(seed)

    # Each solution is composed of num_agents paths
    best_solution: Solution = init_solution(num_agents, num_rewards)
    best_solution.score = evaluate(best_solution, rvalues, distmx, budget)

    no_improve = 0
    kmax = 4 # Number of neighborhoods

    while no_improve < max_no_improve:
        k = 1
        while k <= kmax:
            solution = perturb_solution(best_solution, k)
            solution = local_search(solution, k, rvalues, distmx, budget)

            if solution.score > best_solution.score:
                best_solution = solution.copy()
                no_improve = 0
                k = 1
            else:
                no_improve += 1
                k += 1

    return best_solution


def main(
    num_rewards: int,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    budget: int,
    begin: int = 0,
    end: int = 0,
    max_no_improve: int = 400,
    num_agents: int = 2,
    seed: int = 42,
):
    Solution.set_parameters(begin, end, budget)

    # Matrix of distances between rewards
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    best_solution: Solution = movns(num_agents, num_rewards, rvalues, budget, distmx, max_no_improve, seed)

    bounded_solution = best_solution.get_solution(distmx)
    print(best_solution.score)
    print(best_solution.get_solution_length(distmx))

    return bounded_solution