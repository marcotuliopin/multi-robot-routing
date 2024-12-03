import random
import numpy as np
from sklearn.neighbors import KDTree
from ..shared.utils import get_points_in_range
from deap import tools

def init_individual(icls, num_rewards: list, rpositions: np.ndarray, maxdist:np.ndarray, kdtree: KDTree) -> list:
    if random.random() > 0.8:
        return valid_individual(icls, num_rewards, rpositions, maxdist, kdtree)
    return random_individual(icls, num_rewards)


def random_individual(icls, num_rewards: list) -> list:
    perm1 = np.random.permutation(num_rewards)
    perm2 = np.random.permutation(num_rewards)
    return icls(np.array([perm1, perm2]))


def valid_individual(icls, num_rewards: list, rpositions: np.ndarray, maxdist:np.ndarray, kdtree: KDTree) -> list:
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