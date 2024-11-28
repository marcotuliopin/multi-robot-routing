import numpy as np
from sklearn.neighbors import KDTree


def get_last_valid_idx(path: list, distmx: np.ndarray, budget: int) -> int:
    total_distance = 0
    curr_reward = 0

    valid_idx = 0
    for i in range(len(path)):
        next_reward = path[i]
        total_distance += distmx[curr_reward, next_reward] + distmx[next_reward, 0]
        if total_distance > budget:
            break
        valid_idx = i
        curr_reward = next_reward

    return valid_idx


def get_points_in_range(p: int, rpositions: np.ndarray, maxdist: float, kdtree: KDTree) -> list:
    current_point = rpositions[p]
    indices_within_radius = kdtree.query_radius([current_point], r=maxdist)[0]
    return indices_within_radius
