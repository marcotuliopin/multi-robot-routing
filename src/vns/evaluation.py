import numpy as np
from src.shared.utils import get_last_valid_idx
from utils import interpolate_paths

def maximize_reward(path1: np.ndarray, path2, rvalues: np.ndarray) -> float:
    diff_elements = path1[~np.isin(path1, path2)]
    common_elements = path1[np.isin(path1, path2)]

    return rvalues[diff_elements].sum() + rvalues[common_elements].sum() / 2


def get_distance_between_paths(path1: np.ndarray, path2: np.ndarray, rpositions: np.ndarray) -> np.ndarray:
    path1 = np.concatenate(([0], path1, [0]))
    path2 = np.concatenate(([0], path2, [0]))

    paths = interpolate_paths(path1, path2, rpositions, 1)
    return np.linalg.norm(paths[0] - paths[1], axis=1)


def evaluate(
    path: np.ndarray,
    neighbor: np.ndarray,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    budget: int,
    maxdist: float
) -> float:
    last_idx_path = get_last_valid_idx(path, distmx, budget) + 1
    last_idx_neighbor = get_last_valid_idx(neighbor, distmx, budget) + 1

    path1, path2 = path[:last_idx_path], neighbor[:last_idx_neighbor]

    max_reward = maximize_reward(path1, path2, rvalues)

    distance_between_agents = get_distance_between_paths(path1, path2, rpositions)

    min_distance = distance_between_agents.sum()
    is_solution_valid = np.all(distance_between_agents <= maxdist)

    return max_reward if is_solution_valid else -min_distance
