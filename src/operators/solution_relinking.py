import numpy as np
from ..entities import Solution
from ..evaluation import evaluate


def solution_relinking(
    original_solution1: Solution,
    original_solution2: Solution,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> list[Solution]:
    solution1 = original_solution1.copy()
    solution2 = original_solution2.copy()

    neighbors = []

    for i in range(len(solution1.paths)):
        path1 = solution1.paths[i]
        path2 = solution2.paths[i]

        for j in range(len(path1)):
            if path1[j] == path2[j]:
                continue

            swap_idx = np.where(path1 == path2[j])[0][0]
            path1[j], path1[swap_idx] = path1[swap_idx], path1[j]

            new_solution = solution1.copy()
            new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

            if not any(other.dominates(new_solution) for other in neighbors):
                neighbors.append(new_solution)

    return neighbors
