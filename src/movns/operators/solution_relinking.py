import numpy as np
from ..entities import Solution
from ..evaluation import evaluate


def solution_relinking(
    solution1: Solution,
    solution2: Solution,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> list[Solution]:
    neighbors = []
    num_paths = len(solution1.unbounded_paths)

    for i in range(num_paths):
        path1 = solution1.unbounded_paths[i]
        path2 = solution2.unbounded_paths[i]

        for j in range(len(path1)):
            if path1[j] == path2[j]:
                continue
            swap_idx = np.where(path1 == path2[j])[0][0]
            new_solution = solution1.copy()
            path1[j], path1[swap_idx] = path1[swap_idx], path1[j]
            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)
            if not any(other.dominates(new_solution) for other in neighbors):
                neighbors.append(new_solution)
    return neighbors
