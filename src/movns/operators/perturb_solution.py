import numpy as np
from ..entities import Solution


def perturb_solution(solution: Solution, k: int) -> Solution:
    new_solution = solution.copy()
    new_solution.score = -1

    paths = new_solution.unbounded_paths
    for idx in range(len(paths)):
        if k <= 2:
            paths[idx] = two_opt(paths[idx])
        else:
            paths[idx] = swap_subpaths(paths[idx])

    return new_solution

def two_opt(path: list) -> list:
    i, j = np.random.choice(len(path) - 1, 2, replace=False)
    if i > j:
        i, j = j, i
    new_path = np.concatenate([path[:i], path[i : j + 1][::-1], path[j + 1 :]])
    return new_path

def swap_subpaths(path: np.ndarray) -> np.ndarray:
    new_path = path.copy()
    l = np.random.randint(1, len(path) / 2 - 1)
    i = np.random.randint(0, len(path) - 2 * l)
    j = np.random.randint(i + l, len(path) - l)
    new_path[i : i + l], new_path[j : j + l] = (
        new_path[j : j + l].copy(),
        new_path[i : i + l].copy(),
    )
    return new_path