import numpy as np
from ..entities import Solution

# Time complexity: O(n * m), where n is the number of paths and m is the number of points in each path.
# Space complexity: O(n * m), as it stores the new solution.
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

# Time complexity: O(n), where n is the number of points in the path.
# Space complexity: O(n), as it stores the new path.
def two_opt(path: list) -> list:
    i, j = np.random.choice(len(path) - 1, 2, replace=False)
    if i > j:
        i, j = j, i
    new_path = np.concatenate([path[:i], path[i : j + 1][::-1], path[j + 1 :]])
    return new_path

# Time complexity: O(n), where n is the number of points in the path.
# Space complexity: O(n), as it stores the new path.
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