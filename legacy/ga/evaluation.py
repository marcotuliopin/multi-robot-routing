import numpy as np
from src.shared.utils import get_last_valid_idx
from utils import interpolate_paths


def maximize_reward(ind: list, rvalues: np.ndarray, last_idxs: list) -> float:
    visited = set()
    fitness = 0

    for path, last_idx in zip(ind, last_idxs):

        for reward in path[:last_idx]:
            if reward not in visited:
                fitness += rvalues[reward]
                visited.add(reward)

    return fitness


def minimize_distance(ind: list, distmx: np.ndarray, last_idxs: list) -> float:
    total_distance = 0

    for path, last_idx in zip(ind, last_idxs):
        prev = 0
        partial_distance = 0

        for curr in path[:last_idx]:
            partial_distance += distmx[prev, curr]
            prev = curr

        partial_distance = (partial_distance + distmx[prev, 0]) / last_idx
        total_distance += partial_distance

    return total_distance


def is_solution_valid(
    ind: list, rpositions: np.ndarray, maxdist: float, last_idxs: list
) -> float:
    path1, path2 = list(ind[0][: last_idxs[0]]), list(ind[1][: last_idxs[1]])
    path1.append(0)
    path1.insert(0, 0)
    path2.append(0)
    path2.insert(0, 0)

    paths = interpolate_paths(path1, path2, rpositions, 1)

    distances = np.linalg.norm(paths[0] - paths[1], axis=1)
    return not np.any(distances > maxdist)


def evaluate(
    ind: list,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    maxdist: float,
    budget: int,
) -> float:
    last_idxs = [get_last_valid_idx(path, distmx, budget) + 1 for path in ind]

    max_reward = maximize_reward(ind, rvalues, last_idxs)
    min_distance = minimize_distance(ind, distmx, last_idxs)

    if not is_solution_valid(ind, rpositions, maxdist, last_idxs):
        return (0, 500)

    return (max_reward, min_distance)
