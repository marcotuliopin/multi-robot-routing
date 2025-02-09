import numpy as np
from src.evaluation import evaluate
from ..entities import Solution, Neighborhood


def perturb_solution(
    solution: Solution,
    neighborhood: Neighborhood,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    perturb_operator = neighborhood.get_perturbation_operator()
    new_solution = perturb_operator(solution, rpositions)
    new_solution.paths = new_solution.bound_all_paths(new_solution.paths, distmx, rvalues)
    new_solution.score = evaluate(new_solution, rvalues, rpositions, distmx)

    return new_solution