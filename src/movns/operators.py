import random
import numpy as np
from src.movns.entity.Solution import Solution
from src.movns.entity.Solution import Solution
from src.movns.evaluation import evaluate


def init_solution(num_agents, num_rewards: int) -> tuple:
    val = [np.random.permutation(num_rewards) for _ in range(num_agents)]
    return Solution(val)


def perturb_solution(solution: Solution, k: int) -> Solution:
    new_solution = solution.copy()
    new_solution.score = -1

    paths = new_solution.unbounded_paths
    for idx in range(len(paths)):
        if k == 1:
            paths[idx] = cx_individual(paths[idx], paths[idx])[0]
        if k < 2:
            paths[idx] = two_opt(paths[idx])
        else:
            paths[idx] = swap_subpaths(paths[idx])

    return new_solution


def local_search(
    solution: Solution,
    k: int,
    rvalues: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    best_solution = solution.copy()

    better_solution_found = True

    num_paths = len(best_solution.unbounded_paths)
    i = 0
    while better_solution_found or i < num_paths:
        better_solution_found = False
        new_solution = step(best_solution, i % num_paths, k, rvalues, distmx)

        if new_solution.score > best_solution.score:
            best_solution = new_solution
            better_solution_found = True
        else:
            i += 1

    return best_solution


def step(
    solution: Solution,
    agent: int,
    k: int,
    rvalues: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    best_solution = solution.copy()
    path = best_solution.unbounded_paths[agent]

    # Select the operator according to the neighborhood
    if k % 2 == 0:
        op = move_point
    else:
        op = swap_points

    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            new_solution = best_solution.copy()

            new_path = new_solution.unbounded_paths[agent]
            op(new_path, i, j)

            new_solution.score = evaluate(new_solution, rvalues, distmx)

            if new_solution.score > best_solution.score:
                best_solution.unbounded_paths[agent] = new_path
                best_solution.score = new_solution.score

    return best_solution


def move_point(path: np.ndarray, i: int, j: int) -> np.ndarray:
    element = path[i]
    path = np.delete(path, i)
    path = np.insert(path, j, element)
    return path


def swap_points(path: np.ndarray, i: int, j: int) -> np.ndarray:
    path[i], path[j] = path[j], path[i]
    return path


def two_opt(path: list) -> list:
    i, j = np.random.choice(len(path) - 1, 2, replace=False)
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
    new_path[i : i + l], new_path[j : j + l] = (
        new_path[j : j + l].copy(),
        new_path[i : i + l].copy(),
    )
    return new_path
