import random
import numpy as np
from src.vns.evaluation import evaluate


def init_individual(num_rewards: int) -> tuple:
    return np.random.permutation(num_rewards)


def perturb_solution(path: list, k: int) -> tuple:

    if k == 0:
        new_path, _ = cx_individual(path, path)
    elif k == 1:
        i, j = np.random.choice(len(path) - 1, 2, replace=False)
        new_path = two_opt(path, i, j)
    else:
        new_path = swap_subpaths(path)

    return new_path


def local_search(
    path: np.ndarray,
    k: int,
    rvalues: np.ndarray,
    distmx: np.ndarray,
    budget: int,
):
    best_path = path.copy()
    best_score = evaluate(best_path, rvalues, distmx, budget)

    better_solution_found = True

    while better_solution_found:
        better_solution_found = False

        new_path = step(best_path, k, rvalues, distmx, budget)
        score = evaluate(new_path, rvalues, distmx, budget)

        if score > best_score:
            best_path = new_path.copy()
            best_score = score
            better_solution_found = True

    return best_path, best_score


def step(
    path: np.ndarray,
    k: int,
    rvalues: np.ndarray,
    distmx: np.ndarray,
    budget: int
):
    best_path = path.copy()
    best_score = evaluate(path, rvalues, distmx, budget)

    if k % 2 == 0:
        op = move_point
    else:
        op = swap_points
    
    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            new_path = op(path, i, j)
            new_score = evaluate(new_path, rvalues, distmx, budget)

            if new_score > best_score:
                best_path = new_path.copy()
                best_score = new_score
            
    return best_path


def move_point(path: np.ndarray, i: int, j: int) -> np.ndarray:
    new_path = path.copy()
    element = new_path[i]
    new_path = np.delete(new_path, i)
    new_path = np.insert(new_path, j, element)
    return new_path


def swap_points(path: np.ndarray, i: int, j: int) -> np.ndarray:
    new_path = path.copy()
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def two_opt(path: list, i: int, j: int) -> list:
    if i > j:
        i, j = j, i
    new_path = np.concatenate([path[:i], path[i : j + 1][::-1], path[j + 1 :]])
    return new_path


def cx_individual(ind1: np.ndarray, ind2: np.ndarray) -> tuple:
    size = len(ind1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    p1, p2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
    
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    for i in range(cxpoint1, cxpoint2):
        temp1 = ind1[i]
        temp2 = ind2[i]

        ind1[i], ind2[i] = temp2, temp1
        
        ind1[p1[temp2]] = temp1
        ind2[p2[temp1]] = temp2
        
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


def swap_subpaths(path: np.ndarray) -> np.ndarray:
    new_path = path.copy()
    l = np.random.randint(1, len(path) / 2 - 1)
    i = np.random.randint(0, len(path) - 2 * l)
    j = np.random.randint(i + l, len(path) - l)
    new_path[i : i + l], new_path[j : j + l] = new_path[j : j + l].copy(), new_path[i : i + l].copy()
    return new_path