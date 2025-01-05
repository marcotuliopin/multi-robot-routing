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

    num_paths = len(solution.unbounded_paths)
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

    for i in range(len(solution.unbounded_paths[agent])):
        for j in range(i + 1, len(solution.unbounded_paths[agent])):
            new_solution = solution.copy()

            new_path = new_solution.unbounded_paths[agent]
            new_path[:] = op(new_path, i, j)

            if k == 2 or k == 5:
                for a in range(len(solution.unbounded_paths)):
                    if a != agent:
                        new_path = new_solution.unbounded_paths[a]
                        new_path[:] = op(new_path, i, j)

            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

            # Dominated and non-dominated solutions are stored together and separeted during the archive update
            neighbors.append(new_solution)

    return neighbors


def move_point(path: np.ndarray, i: int, j: int) -> np.ndarray:
    element = path[i]
    path = np.delete(path, i)
    path = np.insert(path, j, element)
    return path


def swap_points(path: np.ndarray, i: int, j: int) -> np.ndarray:
    path[i], path[j] = path[j], path[i]
    return path