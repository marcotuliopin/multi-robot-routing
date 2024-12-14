import numpy as np
from src.shared.utils import get_last_valid_idx


def maximize_reward(path: np.ndarray, rvalues: np.ndarray) -> float:
    return rvalues[path].sum()


def evaluate(
    path: np.ndarray,
    rvalues: np.ndarray,
    distmx: np.ndarray,
    budget: int,
) -> float:
    last_idx_path = get_last_valid_idx(path, distmx, budget) + 1
    path = path[:last_idx_path]

    max_reward = maximize_reward(path, rvalues)
    return max_reward