import numpy as np
from sklearn.neighbors import KDTree
from utils import get_last_valid_idx


def minimize_distance(ind: list, distmx: np.ndarray) -> float:
    total_distance = 0
    curr_reward = 0

    for next_reward in ind:
        total_distance += distmx[curr_reward, next_reward]
        curr_reward = next_reward

    return total_distance


def maximize_reward(ind: list, rpositions: np.ndarray, weights: np.ndarray, lastidx: int, kdtree: KDTree, maxdist: float) -> float:
    reward = 0
    visited = set()
    
    for i in range(lastidx + 1):
        curr = ind[i]

        if curr not in visited:
            reward += weights[curr]
            visited.add(curr)

        indices_within_radius = kdtree.query_radius([rpositions[curr]], r=maxdist)[0]
        best_neighbor = max(indices_within_radius, key=lambda idx: weights[idx])

        if best_neighbor not in visited:
            reward += weights[best_neighbor]
            visited.add(best_neighbor)

    return reward


def evaluate(ind: list, distmx: np.ndarray, rpositions: np.ndarray, weights: np.ndarray, kdtree: KDTree, maxdist: float, budget: int) -> float:
    lastidx = get_last_valid_idx(ind, distmx, budget)

    min_distance = minimize_distance(ind, distmx)
    max_reward = maximize_reward(ind, rpositions, weights, lastidx, kdtree, maxdist)

    return max_reward, min_distance


def calculate_mutation_probability(generation: int, max_generations: int, initial_prob: float, decay_rate: float) -> float:
    return initial_prob * np.exp(-decay_rate * (generation / max_generations))