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


def move_point(solution: Solution, agent: int) -> np.ndarray:
    path = solution.paths[agent]
    neighbors = []

    positive_indices = np.where(path > 0)[0]

    for i in range(len(positive_indices)):
        for j in range(i + 1, len(positive_indices)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            idx1 = positive_indices[i]
            idx2 = positive_indices[j]

            if idx2 == len(positive_indices) - 1:
                # The new position is the last point
                new_path[idx1] = new_path[idx2]
                # The last point is the middle point between the two points
                new_path[idx2] = new_path[idx1] + (new_path[idx2 - 1] - new_path[idx1]) / 2
            else:
                # The new position is the middle point between the two points
                new_path[idx1] = new_path[idx2] + (new_path[idx2 + 1] - new_path[idx2]) / 2

            neighbors.append(new_solution)

    return neighbors


def swap_points(solution: Solution, agent: int) -> np.ndarray:
    path = solution.paths[agent]
    neighbors = []

    positive_indices = np.where(path > 0)[0]

    for i in range(len(positive_indices)):
        for j in range(i + 1, len(positive_indices)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            idx1 = positive_indices[i]
            idx2 = positive_indices[j]
            new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]

            neighbors.append(new_solution)

    return neighbors


def swap_subpaths(path: np.ndarray) -> np.ndarray:
    neighbors = []

    l = np.random.randint(1, len(path) // 2)

    for i in range(len(path) - 2 * l):
        for j in range(i + l, len(path) - l):
            new_path = path.copy()

            new_path[i : i + l], new_path[j : j + l] = (
                new_path[j : j + l].copy(),
                new_path[i : i + l].copy(),
            )
            neighbors.append(new_path)
            
    return neighbors


def invert_single_point(path: np.ndarray) -> np.ndarray:
    neighbors = []

    for i in range(len(path)):
        new_solution = path.copy()
        new_solution[i] = -new_solution[i]
        neighbors.append(new_solution)
    
    return neighbors


def invert_multiple_points(path: np.ndarray) -> np.ndarray:
    neighbors = []
    
    for i in range(1, len(path) + 1):
        new_solution = path.copy()
        
        idxs = np.random.choice(len(path), i, replace=False)
        new_solution[idxs] = -new_solution[idxs]
        neighbors.append(new_solution)
    
    return neighbors


def swap_subpaths_all_paths(solution: Solution, l: int) -> Solution:
    new_solution = solution.copy()
    num_paths = len(solution.paths)
    
    for i in range(num_paths):
        path = new_solution.paths[i]
        positive_indices = np.where(path > 0)[0]

        if len(positive_indices) < 2 * l:
            continue

        idx1 = positive_indices[:l]
        idx2 = positive_indices[-l:]

        new_solution.paths[i][idx1], new_solution.paths[i][idx2] = new_solution.paths[i][idx2].copy(), new_solution.paths[i][idx1].copy()

    return new_solution

# def get_sorted_indices(path: np.ndarray) -> np.ndarray:
#     positive_indices = np.where(path > 0)[0]
#     return positive_indices[np.argsort(path[positive_indices])]
