import numpy as np
from utils import get_last_valid_idx
from deap import tools


def init_individual(icls, clusters: list) -> list:
    perm1 = []
    perm2 = []

    visited_points = set()

    for cluster_points in clusters:
        unique_points = [p for p in cluster_points if p not in visited_points]

        if unique_points:
            shuffled_points1 = np.random.permutation(unique_points)
            shuffled_points2 = np.random.permutation(unique_points)

            perm1.extend(shuffled_points1)
            perm2.extend(shuffled_points2)

            visited_points.update(unique_points)

    genes = np.array([perm1, perm2])

    num_columns = genes.shape[1]
    shuffled_indices = np.random.permutation(num_columns)

    genes = genes[:, shuffled_indices]

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


def evaluate(ind: list, gen: int, ngen: int, rvalues: np.ndarray, distmx: np.ndarray, maxdist: float) -> float:
    max_reward = maximize_reward(ind, rvalues)
    min_distance = minimize_distance(ind, distmx)
    min_distance_between_agents = minimize_distance_between_agents(ind, distmx)

    penalize_dist = penalize_distance(ind, distmx, maxdist)

    # Add a big penalty if the distance is too high after half of the generations.
    if penalize_dist and gen > ngen // 2:
        return (-1000, min_distance + 5000, min_distance_between_agents + 5000)

    # Add a small penalty if the distance is too high. This lets the algorithm explore more.
    elif penalize_dist:
        reward_penalty = 500 * (1 + gen / ngen)
        distance_penalty = 1000 * (1 + gen / ngen)
        return (max_reward - reward_penalty, min_distance + distance_penalty, min_distance_between_agents + distance_penalty)

    return (max_reward, min_distance, min_distance_between_agents)


def maximize_reward(ind: list, rvalues: np.ndarray) -> float:
    visited = set()
    fitness = 0

    for path in ind:
        last_idx = get_last_valid_idx(path)

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


def minimize_distance_between_agents(ind: list, distmx: np.ndarray) -> float:
    last_idx1 = get_last_valid_idx(ind[0])
    last_idx2 = get_last_valid_idx(ind[1])
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


def penalize_distance(ind: list, distmx: np.ndarray, maxdist: float) -> float:
    last_idx1 = get_last_valid_idx(ind[0])
    last_idx2 = get_last_valid_idx(ind[1])
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