import numpy as np
from ..entities import Solution
from ..evaluation import evaluate


def local_search(
    solution: Solution,
    neighborhood: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    neighbors = []

    for i in range(len(solution.paths)):
        neighbors.extend(step(solution, i, neighborhood))

    for neighbor in neighbors:
        neighbor.paths = neighbor.bound_all_paths(neighbor.paths, distmx, rvalues)
        neighbor.score = evaluate(neighbor, rvalues, rpositions, distmx)
            
    return neighbors


def step(solution: Solution, agent: int, neighborhood: int) -> Solution:
    operators = {
        0: move_point(solution, agent),
        1: invert_single_point(solution, agent),
        2: swap_points(solution, agent),
        3: invert_multiple_points(solution, agent),
        4: swap_subpaths(solution, agent),
        5: move_point(solution, agent),
        6: invert_single_point(solution, agent),
        7: swap_points(solution, agent),
        8: invert_multiple_points(solution, agent),
        9: swap_subpaths(solution, agent),
    }

    return operators[neighborhood]


def move_point(solution: Solution, agent: int) -> list[Solution]:
    path = solution.paths[agent]
    neighbors = []

    positive_indices = np.where(path > 0)[0]

    for i in range(len(positive_indices)):
        for j in range(i + 1, len(positive_indices)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            idx1 = positive_indices[i]
            idx2 = positive_indices[j]

            if j == len(positive_indices) - 1:
                # The new position is the last point
                new_path[idx1] = new_path[idx2]
                # The last point is the middle point between the two points
                new_path[idx2] = new_path[idx1] + (new_path[positive_indices[j - 1]] - new_path[idx1]) / 2
            else:
                # The new position is the middle point between the two points
                new_path[idx1] = new_path[idx2] + (new_path[positive_indices[j + 1]] - new_path[idx2]) / 2

            neighbors.append(new_solution)

    return neighbors


def swap_points(solution: Solution, agent: int) -> list[Solution]:
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


def swap_subpaths(solution: Solution, agent: int) -> list[Solution]:
    path = solution.paths[agent]
    neighbors = []

    l = np.random.randint(1, len(path) // 2)

    for i in range(len(path) - 2 * l):
        for j in range(i + l, len(path) - l):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            new_path[i : i + l], new_path[j : j + l] = (
                new_path[j : j + l].copy(),
                new_path[i : i + l].copy(),
            )

            neighbors.append(new_solution)
            
    return neighbors


def invert_single_point(solution: Solution, agent: int) -> list[Solution]:
    path = solution.paths[agent]
    neighbors = []

    for i in range(len(path)):
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]

        new_path[i] = -new_path[i]

        neighbors.append(new_solution)
    
    return neighbors


def invert_multiple_points(solution: Solution, agent: int) -> list[Solution]:
    path = solution.paths[agent]
    neighbors = []
    
    for i in range(1, len(path) + 1):
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        
        idxs = np.random.choice(len(new_path), i, replace=False)
        new_path[idxs] = -new_path[idxs]

        neighbors.append(new_solution)
    
    return neighbors


# def get_sorted_indices(path: np.ndarray) -> np.ndarray:
#     positive_indices = np.where(path > 0)[0]
#     return positive_indices[np.argsort(path[positive_indices])]
