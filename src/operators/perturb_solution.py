import numpy as np
from src.evaluation import evaluate
from ..entities import Solution


def perturb_solution(
    solution: Solution,
    neighborhood: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    if neighborhood % 2 == 0:
        new_solution = two_opt(solution)
    else:
        new_solution = swap_subpaths(solution)

    new_solution.paths = new_solution.bound_all_paths(new_solution.paths, distmx, rvalues)
    new_solution.score = evaluate(new_solution, rvalues, rpositions)

    return new_solution


def two_opt(solution: Solution) -> list:
    new_solution = solution.copy()
    new_paths = new_solution.paths

    i, j = np.random.choice(len(new_paths[0]) - 1, 2, replace=False)
    if i > j:
        i, j = j, i

    for new_path in new_paths:
        new_path = np.concatenate([new_path[:i], new_path[i : j + 1][::-1], new_path[j + 1 :]])

    return new_solution


def swap_subpaths(solution: Solution) -> np.ndarray:
    new_solution = solution.copy()
    new_paths = new_solution.paths

    l = np.random.randint(1, len(new_paths[0]) // 2)
    i = np.random.randint(0, len(new_paths[0]) - 2 * l)
    j = np.random.randint(i + l, len(new_paths[0]) - l)

    for new_path in new_paths:
        new_path[i : i + l], new_path[j : j + l] = (
            new_path[j : j + l].copy(),
            new_path[i : i + l].copy(),
        )

    return new_solution
