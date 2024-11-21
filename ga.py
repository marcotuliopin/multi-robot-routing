import numpy as np
from sklearn.neighbors import KDTree
from utils import get_last_valid_idx, get_points_in_range
from deap import tools


def init_individual(icls, num_rewards: list, rpositions: np.ndarray, maxdist:np.ndarray, kdtree: KDTree) -> list:
    perm1 = []
    perm2 = []

    available =  set(range(num_rewards))

    for i in np.random.permutation(num_rewards):
        points_in_range = get_points_in_range(i, rpositions, maxdist, kdtree)
        available_points_in_range = [p for p in points_in_range if p in available]

        if available_points_in_range:
            shuffled_points1 = np.random.permutation(available_points_in_range)
            shuffled_points2 = np.random.permutation(available_points_in_range)

            perm1.extend(shuffled_points1)
            perm2.extend(shuffled_points2)

            available.difference_update(available_points_in_range)

    genes = np.array([perm1, perm2])

    return icls(genes)


def mut_individual(individual: list, indpb: float) -> tuple:
    path1, path2 = individual
    tools.mutShuffleIndexes(path1, indpb=indpb)
    tools.mutShuffleIndexes(path2, indpb=indpb)
    return (individual,)


def cx_individual(ind1: list, ind2: list) -> tuple:
    tools.cxPartialyMatched(ind1[0], ind2[0])
    tools.cxPartialyMatched(ind1[1], ind2[1])
    return ind1, ind2


def evaluate(ind: list, gen: int, ngen: int, rvalues: np.ndarray, distmx: np.ndarray, maxdist: float, budget: int) -> float:
    last_valid_idxs = [get_last_valid_idx(path, distmx, budget) for path in ind]

    max_reward = maximize_reward(ind, rvalues, last_valid_idxs)
    min_distance = minimize_distance(ind, distmx)
    min_distance_between_agents = minimize_distance_between_agents(ind, distmx, last_valid_idxs)

    penalize_dist = penalize_distance(ind, distmx, maxdist, last_valid_idxs)

    if penalize_dist:
        # Add a big penalty if the distance is too high after half of the generations.
        if gen > ngen // 2:
            return (-1000, min_distance + 5000, min_distance_between_agents + 5000)

        # Add a small penalty if the distance is too high. This lets the algorithm explore more.
        reward_penalty = 500 * (1 + gen / ngen)
        distance_penalty = 1000 * (1 + gen / ngen)
        return (max_reward - reward_penalty, min_distance + distance_penalty, min_distance_between_agents + distance_penalty)

    return (max_reward, min_distance, min_distance_between_agents)


def maximize_reward(ind: list, rvalues: np.ndarray, last_valid_idxs: list) -> float:

    visited = set()
    fitness = 0

    for p, path in enumerate(ind):
        last_idx = last_valid_idxs[p]

        for i in range(last_idx + 1):
            next_reward = path[i]

            if not next_reward in visited:
                fitness += rvalues[next_reward]
                visited.add(next_reward)

    return fitness


def minimize_distance(ind: list, distmx: np.ndarray) -> float:
    total_distance = 0
    for path in ind:
        prev = 0
        for curr in path:
            total_distance += distmx[prev, curr]
            prev = curr

    return total_distance


def minimize_distance_between_agents(ind: list, distmx: np.ndarray, last_valid_idxs: list) -> float:
    last_idx1, last_idx2 = last_valid_idxs
    min_last_idx = min(last_idx1, last_idx2)

    full_distance = 0

    for p1, p2 in zip(ind[0][:min_last_idx], ind[1][:min_last_idx]):
        full_distance += distmx[p1, p2]

    if last_idx1 > last_idx2:
        for p1 in ind[0][min_last_idx:last_idx1]:
            full_distance += distmx[p1, p2]
    elif last_idx2 > last_idx1:
        for p2 in ind[1][min_last_idx:last_idx2]:
            full_distance += distmx[p1, p2]

    return full_distance


def penalize_distance(ind: list, distmx: np.ndarray, maxdist: float, last_valid_idxs: list) -> float:
    last_idx1, last_idx2 = last_valid_idxs
    min_last_idx = min(last_idx1, last_idx2)

    for p1, p2 in zip(ind[0][:min_last_idx+1], ind[1][:min_last_idx+1]):
        if distmx[p1, p2] > maxdist:
            return True

    if last_idx1 > last_idx2:
        for p1 in ind[0][min_last_idx+1:last_idx1]:
            if distmx[p1, ind[1][min_last_idx - 1]] > maxdist:
                return True

    elif last_idx2 > last_idx1:
        for p2 in ind[1][min_last_idx+1:last_idx2]:
            if distmx[ind[0][min_last_idx - 1], p2] > maxdist:
                return True

    return False