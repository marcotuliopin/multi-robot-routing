import numpy as np
from ..entities import Solution, Neighborhood
from ..evaluation import evaluate


def local_search(
    solution: Solution,
    neighborhood: Neighborhood,
    neighborhood_id: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> Solution:
    neighbors = []

    for agent in range(len(solution.paths)):
        local_search_operator = neighborhood.get_local_search_operator(neighborhood_id)
        neighbors.extend(local_search_operator(solution, agent))

    for neighbor in neighbors:
        neighbor.paths = neighbor.bound_all_paths(neighbor.paths, distmx, rvalues)
        neighbor.score = evaluate(neighbor, rvalues, rpositions)
            
    return neighbors