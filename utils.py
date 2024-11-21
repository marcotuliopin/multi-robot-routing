import numpy as np
from deap import tools
from sklearn.neighbors import KDTree


def add_neighbor_values(rvalues: np.ndarray, rpositions: np.ndarray, kdtree: KDTree, maxdist: np.ndarray) -> np.ndarray:
    rvalues = np.array(rvalues)
    for i, pos in enumerate(rpositions):
        neighbors = kdtree.query_radius([pos], r=maxdist)[0]
        rvalues[i] = rvalues[i] + max(rvalues[neighbors])
    return rvalues


def get_last_valid_idx(path: list, distmx: np.ndarray, budget: int) -> int:
    total_distance = 0
    curr_reward = 0

    valid_idx = 0
    for i in range(len(path)):
        next_reward = path[i]
        total_distance += distmx[curr_reward, next_reward]
        if total_distance > budget:
            break
        valid_idx = i
        curr_reward = next_reward

    return valid_idx


def assign_crowding_dist(population):
    tools.sortNondominated(population, len(population), first_front_only=False)

    for front in population:
        front_size = len(front)
        if front_size > 0:
            front[0].fitness.crowding_dist = float('inf')
            front[-1].fitness.crowding_dist = float('inf')

            for i in range(len(front[0].fitness.values)):
                crowd = sorted([(ind.fitness.values, ind) for ind in front], key=lambda element: element[0][i])
                
                if crowd[-1][0][i] == crowd[0][0][i]:
                    continue
                
                norm = float(crowd[-1][0][i] - crowd[0][0][i])
                
                for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
                    dist = (next[0][i] - prev[0][i]) / norm
                    cur[1].fitness.crowding_dist += dist


def calculate_mutation_probability(generation: int, max_generations: int, initial_prob: float, decay_rate: float) -> float:
    return initial_prob * np.exp(-decay_rate * (generation / max_generations))


def get_points_in_range(p: int, rpositions: np.ndarray, maxdist: float, kdtree: KDTree) -> list:
    current_point = rpositions[p]
    indices_within_radius = kdtree.query_radius([current_point], r=maxdist)[0]
    return indices_within_radius
