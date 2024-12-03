import random
import numpy as np
from scipy.spatial.distance import cdist
from src.shared.utils import get_last_valid_idx, get_path_length
from src.vns.evaluation import evaluate
from src.vns.operators import init_individual, local_search, perturb_solution


def vns(
    num_rewards: int,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    maxdist: float,
    budget: int,
    distmx: np.ndarray,
    max_iters: int = 1000,
    max_no_improve: int = 100,
    seed: int = 42,
):
    random.seed(seed)
    best_path1, best_path2 = init_individual(num_rewards)

    best_score1 = evaluate(best_path1, best_path2, rvalues, rpositions, distmx, budget, maxdist)
    best_score2 = evaluate(best_path2, best_path1, rvalues, rpositions, distmx, budget, maxdist)
    best_score = best_score1 + best_score2

    no_improve = 0

    for _ in range(max_iters):
        for _ in range(1, 4):  # Define 3 neighborhoods
            path1, path2 = perturb_solution(best_path1, best_path2)
            path1, path2, score = local_search(path1, path2, rvalues, rpositions, distmx, maxdist, budget)

            # Update the best solution
            if score > best_score:
                best_path1, best_path2 = path1, path2
                best_score = score
                no_improve = 0
                break
            else:
                no_improve += 1  # No improvement in this iteration
            
            print(best_score)

    return best_path1, best_path2, best_score


def main(
    num_rewards: int,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    maxdist: float,
    budget: int,
    max_iters: int = 3000,
    max_no_improve: int = 100,
    seed: int = 42,
):
    distmx = cdist(rpositions, rpositions, metric="euclidean")

    best_path1, best_path2, best_score = vns(num_rewards, rpositions, rvalues, maxdist, budget, distmx, max_iters, max_no_improve, seed)

    last_idx1 = get_last_valid_idx(best_path1, distmx, budget) + 1
    path1 = list(best_path1[:last_idx1])

    last_idx2 = get_last_valid_idx(best_path2, distmx, budget) + 1
    path2 = list(best_path2[:last_idx2])

    path1.append(0)
    path1.insert(0, 0)
    path2.append(0)
    path2.insert(0, 0)

    print(best_score)
    print(get_path_length(path1, distmx), get_path_length(path2, distmx))

    return path1, path2
