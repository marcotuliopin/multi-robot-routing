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