import random
import numpy as np
from src.movns.entity.Solution import Solution
from src.movns.evaluation import evaluate


def init_solution(num_agents, num_rewards: int) -> tuple:
    val = [np.random.permutation(num_rewards) for _ in range(num_agents)]
    return Solution(val)


def perturb_solution(solution: Solution, k: int) -> Solution:
    new_solution = solution.copy()

    paths = new_solution.unbound_solution
    for idx in range(len(paths)):
        if k < 2:
            i, j = np.random.choice(len(paths[idx]) - 1, 2, replace=False)
            paths[idx] = two_opt(paths[idx], i, j)
        else:
            paths[idx] = swap_subpaths(paths[idx])

    return new_solution


def local_search(
    solution: Solution,
    k: int,
    rvalues: np.ndarray,
    distmx: np.ndarray,
    budget: int,
) -> Solution:
    best_solution = solution.copy()

    better_solution_found = True

    while better_solution_found:
        better_solution_found = False

        num_agents = len(best_solution.unbound_solution)
        for i in range(num_agents):
            new_solution = step(best_solution, i, k, rvalues, distmx, budget)

            if new_solution.score > best_solution.score:
                best_solution = new_solution
                better_solution_found = True

    return best_solution


def step(
    solution: Solution,
    agent: int,
    k: int,
    rvalues: np.ndarray,
    distmx: np.ndarray,
    budget: int
) -> Solution:
    best_solution = solution.copy()
    path = best_solution.unbound_solution[agent] 

    # Select the operator according to the neighborhood
    if k % 2 == 0:
        op = move_point
    else:
        op = swap_points

    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            new_path = op(path, i, j)

            new_solution = best_solution.copy()
            new_solution.unbound_solution[agent] = new_path
            new_solution.score = evaluate(new_path, rvalues, distmx, budget)

            if new_solution.score > best_solution.score:
                best_solution.unbound_solution[agent] = new_path
                best_solution.score = new_solution.score
            
    return best_solution


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