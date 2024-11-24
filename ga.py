import random
import numpy as np
from sklearn.neighbors import KDTree
from utils import get_last_valid_idx, get_points_in_range, interpolate_paths
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
    if random.random() < .5:
        tools.cxPartialyMatched(path1, path2)
    else:
        tools.mutShuffleIndexes(path1, indpb=indpb)
        tools.mutShuffleIndexes(path2, indpb=indpb)
    return (individual,)


def cx_individual(ind1: list, ind2: list) -> tuple:
    tools.cxPartialyMatched(ind1[0], ind2[0])
    tools.cxPartialyMatched(ind1[1], ind2[1])
    return ind1, ind2


def cx_partialy_matched(ind1: list, ind2: list) -> tuple:
    size = ind1.shape[1]
    
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    
    for row in range(ind1.shape[0]):
        p1, p2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
        
        for i in range(size):
            p1[ind1[row][i]] = i
            p2[ind2[row][i]] = i

        for i in range(cxpoint1, cxpoint2):
            temp1 = ind1[row, i]
            temp2 = ind2[row, i]

            ind1[row, i], ind2[row, i] = temp2, temp1
            
            ind1[row, p1[temp2]] = temp1
            ind2[row, p2[temp1]] = temp2

            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


def evaluate(ind: list, rvalues: np.ndarray, rpositions: np.ndarray, distmx: np.ndarray, maxdist: float, budget: int) -> float:
    last_idxs = [get_last_valid_idx(path, distmx, budget) + 1 for path in ind]

    max_reward = maximize_reward(ind, rvalues, last_idxs)
    min_distance = minimize_distance(ind, distmx, last_idxs)

    penalize_dist = penalize_distance(ind, rpositions, maxdist, last_idxs)
    if penalize_dist:
        return (0, np.inf)

    return (max_reward, min_distance)


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


def penalize_distance(ind: list, rpositions: np.ndarray, maxdist: float, last_idxs: list) -> float:
    paths = interpolate_paths(ind[0][:last_idxs[0]], ind[1][:last_idxs[1]], rpositions, 60)

    distances = np.linalg.norm(paths[0] - paths[1], axis=1)
    return np.any(distances > maxdist)