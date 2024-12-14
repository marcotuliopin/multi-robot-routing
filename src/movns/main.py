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
    initial_solution: Solution = init_solution(num_agents, num_rewards)
    initial_solution.score = evaluate(initial_solution, rvalues, distmx, budget)

    no_improve = 0
    kmax = 4 # Number of neighborhoods
    archive = [initial_solution]


    while no_improve < max_no_improve:
        k = 1
        while k <= kmax:
            path = perturb_solution(best_path, k)
            path, score = local_search(path, k, rvalues, distmx, budget)

            if score > best_score:
                best_path = path.copy()
                best_score = score
                no_improve = 0
                k = 1
            else:
                no_improve += 1
                k += 1

    return best_path, best_score


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

    best_paths, best_score = movns(num_agents, num_rewards, rvalues, budget, distmx, max_no_improve, seed)

    last_idx1 = get_last_valid_idx(best_paths[0], distmx, budget) + 1
    path1 = list(best_paths[0][:last_idx1])

    path1.append(begin)
    path1.insert(0, end)

    print(best_score)
    print(get_path_length(path1, distmx))

    return path1