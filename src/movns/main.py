import random
import numpy as np
from scipy.spatial.distance import cdist
from src.movns.evaluation import evaluate
from src.movns.operators import init_solution, local_search, perturb_solution
from src.movns.entity.Solution import Solution

def movns(
    num_agents: int,
    num_rewards: int,
    rvalues: np.ndarray,
    distmx: np.ndarray,
    max_no_improve: int,
    seed: int,
):
    random.seed(seed)

    # Each solution is composed of num_agents paths
    best_solution: Solution = init_solution(num_agents, num_rewards)
    best_solution.score = evaluate(best_solution, rvalues, distmx)

    no_improve = 0
    kmax = 4 # Number of neighborhoods

    while no_improve < max_no_improve:
        k = 1
        while k <= kmax:
            solution = perturb_solution(best_solution, k)
            solution.score = evaluate(solution, rvalues, distmx)

            solution = local_search(solution, k, rvalues, distmx)
            print(best_solution.score)

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
    max_no_improve: int = 200,
    num_agents: int = 2,
    seed: int = 42,
):
    Solution.set_parameters(begin, end, budget)
    print(Solution._BUDGET)

    # Matrix of distances between rewards
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    best_solution: Solution = movns(num_agents, num_rewards, rvalues, distmx, max_no_improve, seed)

    bounded_paths = best_solution.get_solution_paths(distmx)
    print(best_solution.score)
    print(best_solution.get_solution_length(distmx))
    print(best_solution.get_solution_paths(distmx))

    return bounded_paths