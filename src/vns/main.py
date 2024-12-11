import random
import numpy as np
from scipy.spatial.distance import cdist
from src.shared.utils import get_last_valid_idx, get_path_length
from src.vns.evaluation import evaluate
from src.vns.operators import init_individual, local_search, perturb_solution


def vns(
    num_rewards: int,
    rvalues: np.ndarray,
    budget: int,
    distmx: np.ndarray,
    max_no_improve: int = 100,
    seed: int = 42,
):
    random.seed(seed)
    best_path = init_individual(num_rewards)
    best_score = evaluate(best_path, rvalues, distmx, budget)

    no_improve = 0
    kmax = 4

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
    max_no_improve: int = 400,
    seed: int = 42,
):
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    best_path1, best_score = vns(num_rewards, rvalues, budget, distmx, max_no_improve, seed)

    last_idx1 = get_last_valid_idx(best_path1, distmx, budget) + 1
    path1 = list(best_path1[:last_idx1])

    path1.append(0)
    path1.insert(0, 0)

    print(best_score)
    print(get_path_length(path1, distmx))

    return path1
