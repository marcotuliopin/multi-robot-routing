import numpy as np
from ..entities import Solution
from ..evaluation import evaluate

# Time complexity: O(n * m^2), where n is the number of paths and m is the number of points in each path.
# Space complexity: O(n), as it stores the neighbors.
def local_search(
    solution: Solution,
    k: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    neighbors = []

    num_paths = len(solution.unbounded_paths)
    for i in range(num_paths):
        neighbors.extend(step(solution, i, k, rvalues, rpositions, distmx))
            
    return neighbors

# Time complexity: O(m^2), where m is the number of points in the path.
# Space complexity: O(n), as it stores the neighbors.
def step(
    solution: Solution,
    agent: int,
    k: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    neighbors = []

    # Select the operator according to the neighborhood
    if k == 0 or k == 3:
        op = move_point
    else:
        op = swap_points

    len_path = len(solution.unbounded_paths[agent])
    for i in range(len_path):
        for j in range(i + 1, len_path):
            new_solution = solution.copy()

            new_path = new_solution.unbounded_paths[agent]
            new_path[:] = op(new_path, i, j)

            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

            # Dominated and non-dominated solutions are stored together and separeted during the archive update
            # if not any(other.dominates(new_solution) for other in neighbors):
            neighbors.append(new_solution)
    return neighbors

# Time complexity: O(m), where m is the number of points in the path.
# Space complexity: O(m), as it stores the modified path.
def move_point(path: np.ndarray, i: int, j: int) -> np.ndarray:
    element = path[i]
    path = np.delete(path, i)
    path = np.insert(path, j, element)
    return path

# Time complexity: O(1), as it swaps two points in the path.
# Space complexity: O(1), as it uses a constant amount of additional space.
def swap_points(path: np.ndarray, i: int, j: int) -> np.ndarray:
    path[i], path[j] = path[j], path[i]
    return path
