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
        if k <= 2:
            paths[idx] = two_opt(paths[idx])
        else:
            paths[idx] = swap_subpaths(paths[idx])

    return new_solution


def local_search(
    solution: Solution,
    k: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    neighbors = []

    num_paths = len(solution.unbounded_paths)
    for i in range(num_paths):
        neighbors.extend(step(solution, i, k, rvalues, rpositions, distmx))

    return neighbors


def step(
    solution: Solution,
    agent: int,
    k: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    neighbors = []

    # Select the operator according to the neighborhood
    if k % 2 == 0:
        op = move_point
    else:
        op = swap_points

    for i in range(len(solution.unbounded_paths[agent])):
        for j in range(i + 1, len(solution.unbounded_paths[agent])):
            new_solution = solution.copy()

            new_path = new_solution.unbounded_paths[agent] # TODO: Optimize to avoid operations outside budget
            new_path[:] = op(new_path, i, j)

            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

            if not any(neigh.dominates(new_solution) for neigh in neighbors):
                neighbors = [neigh for neigh in neighbors if not new_solution.dominates(neigh)]
                neighbors.append(new_solution)

    return neighbors


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


def solution_relinking(
    solution1: Solution,
    solution2: Solution,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> list[Solution]:
    neighbors = []
    num_paths = len(solution1.unbounded_paths)
    new_solution = solution1.copy()

    for i in range(num_paths):
        path1 = new_solution.unbounded_paths[i]
        path2 = solution2.unbounded_paths[i]

        for j in range(len(path1)):
            if path1[j] == path2[j]:
                continue
            swap_idx = np.where(path1 == path2[j])[0][0]
            path1[j], path1[swap_idx] = path1[swap_idx], path1[j]
            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

            if not any(neigh.dominates(new_solution) for neigh in neighbors):
                neighbors = [neigh for neigh in neighbors if not new_solution.dominates(neigh)]
                neighbors.append(new_solution)

    return neighbors
