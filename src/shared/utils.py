import numpy as np
from sklearn.neighbors import KDTree


def get_last_valid_idx(path: list, distmx: np.ndarray, budget: int) -> int:
    total_distance = 0
    curr_reward = 0

    for i in range(len(path)):
        next_reward = path[i]
        total_distance += distmx[curr_reward, next_reward]

        if total_distance + distmx[next_reward, 0] > budget:
            return i - 1

        curr_reward = next_reward

    return len(path) - 1


def get_points_in_range(p: int, rpositions: np.ndarray, maxdist: float, kdtree: KDTree) -> list:
    current_point = rpositions[p]
    indices_within_radius = kdtree.query_radius([current_point], r=maxdist)[0]
    return indices_within_radius


def get_path_length(path: list, distmx: np.ndarray) -> float:
    length = 0
    for i in range(1, len(path)):
        length += distmx[path[i - 1], path[i]]
    return length
