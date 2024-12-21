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
    if k % 2 == 0:
        op = move_point
    else:
        op = swap_points

    for i in range(len(solution.unbounded_paths[agent])):
        for j in range(i + 1, len(solution.unbounded_paths[agent])):
            new_solution = solution.copy()

            new_path = new_solution.unbounded_paths[agent] # TODO: Optimize to avoid operations outside budget
            new_path[:] = op(new_path, i, j)

            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

            # if not any(neigh.dominates(new_solution) for neigh in neighbors):
            #     neighbors = [neigh for neigh in neighbors if not new_solution.dominates(neigh)]
            #     neighbors.append(new_solution)
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