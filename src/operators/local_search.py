import numpy as np
from ..entities import Solution
from ..evaluation import evaluate

def local_search(
    solution: Solution,
    k: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    neighbors = []

    num_paths = len(solution.paths)
    for i in range(num_paths):
        neighbors.extend(step(solution, i, k, rvalues, rpositions, distmx))
            
    return neighbors

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

    positive_indices = np.where(solution.paths[agent] > 0)[0]
    
    for i in range(len(positive_indices)):
        for j in range(i + 1, len(positive_indices)):
            new_solution = solution.copy()

            new_path = new_solution.paths[agent]
            new_path[:] = op(new_path, positive_indices[i], positive_indices[j])

            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

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
