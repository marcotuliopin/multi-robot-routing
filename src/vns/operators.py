import numpy as np
from src.vns.evaluation import evaluate


def init_individual(num_rewards: int) -> tuple:
    path1 = np.random.permutation(num_rewards)
    path2 = np.random.permutation(num_rewards)
    return path1, path2


def two_opt(path: list, i: int = None, j: int = None) -> list:
    if i is None and j is None:
        i, j = np.random.choice(len(path) - 1, 2, replace=False)
        if i > j:
            i, j = j, i

    new_path = np.concatenate([path[:i], path[i:j+1][::-1], path[j+1:]])
    return new_path


def perturb_solution(path1: list, path2: list, k: int = 10) -> tuple:
    new_path1, new_path2 = path1.copy(), path2.copy()

    for _ in range(k):
        new_path1 = two_opt(new_path1)
        new_path2 = two_opt(new_path2)

    return new_path1, new_path2


def step(
    path: list,
    neighbor: list,
    best_score: float,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    maxdist: float,
    budget: int,
):
    new_path = path.copy()
    new_score = best_score

    for i in range(1, len(path) - 1):
        for j in range(i + 1, len(path)) :
            if j - i == 1: 
                continue

            new_path = two_opt(path, i, j)
            new_score = evaluate(new_path, neighbor, rvalues, rpositions, distmx, budget, maxdist)

            if new_score > best_score:
                return new_path, new_score, True

    return path, best_score, False


def local_search(
    path1: list,
    path2: list,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    maxdist: np.ndarray,
    budget: int,
):
    score1 = evaluate(path1, path2, rvalues, rpositions, distmx, budget, maxdist)
    score2 = evaluate(path2, path1, rvalues, rpositions, distmx, budget, maxdist)

    better_solution_found = True
    while better_solution_found:
        better_solution_found = False

        path1, score1, f1 = step(path1, path2, score1, rvalues, rpositions, distmx, maxdist, budget)
        path2, score2, f2 = step(path2, path1, score2, rvalues, rpositions, distmx, maxdist, budget)

        if f1 or f2:
            better_solution_found = True
            break


    return path1, path2, score1 + score2
