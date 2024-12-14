import numpy as np
from src.movns.entity.Solution import Solution


def maximize_reward(path1: np.ndarray, path2, rvalues: np.ndarray) -> float:
    diff_elements = path1[~np.isin(path1, path2)]
    common_elements = path1[np.isin(path1, path2)]

    return rvalues[diff_elements].sum() + rvalues[common_elements].sum() / 2


def evaluate(
    solution: Solution,
    rvalues: np.ndarray,
    distmx: np.ndarray,
) -> float:
    bounded_paths: list[np.ndarray] = solution.get_solution_paths(distmx)
    max_reward = maximize_reward(bounded_paths[0], bounded_paths[1], rvalues) + maximize_reward(bounded_paths[1], bounded_paths[0], rvalues)
    return max_reward
