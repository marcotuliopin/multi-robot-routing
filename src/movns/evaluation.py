import numpy as np
from .entities import Solution
from utils import calculate_rssi


def evaluate(
    solution: Solution,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> float:
    bounded_paths: list[np.ndarray] = solution.get_solution_paths(distmx)
    max_reward = maximize_reward(bounded_paths, rvalues)
    max_communication = calculate_rssi(bounded_paths, rpositions)
    return max_reward, max_communication


def maximize_reward(paths: list[np.ndarray], rvalues: np.ndarray) -> float:
    all_elements = np.concatenate(paths)
    unique_elements, counts = np.unique(all_elements, return_counts=True)
    reward = 0
    for element, count in zip(unique_elements, counts):
        if count == 1:
            reward += rvalues[element]
        else:
            reward += rvalues[element] / count
    return reward


def update_archive(archive: list[Solution], neighbors: list[Solution], archive_max_size: int) -> tuple:
    all_solutions = archive + neighbors

    non_dominated, dominated = get_non_dominated_solutions(all_solutions)
    
    if len(non_dominated) > archive_max_size:
        non_dominated = select_by_crowding_distance(non_dominated, archive_max_size)

    # If there is space left in the archive, add the non-dominated solutions
    selected_dominated = []
    if len(non_dominated) < archive_max_size:
        selected_dominated = select_by_crowding_distance(dominated, min(archive_max_size - len(non_dominated), len(dominated)))

    archive = non_dominated + selected_dominated[:max(0, archive_max_size - len(non_dominated))]

    return archive, non_dominated, selected_dominated


def get_non_dominated_solutions(solutions: list[Solution]) -> list[Solution]:
    non_dominated = []
    dominated = []
    solutions.sort(key=lambda s: s.score)

    for i in range(len(solutions)):
        if any(s.dominates(solutions[i]) for s in solutions):
            dominated.append(solutions[i])
        else:
            non_dominated.append(solutions[i])
        
    return non_dominated, dominated


def select_by_crowding_distance(solutions: list[Solution], k: int) -> list[Solution]:
    assign_crowding_distance(solutions)
    solutions.sort(key=lambda s: s.crowding_distance, reverse=True)
    return solutions[:k]


def assign_crowding_distance(solutions: list[Solution]) -> None:
    num_solutions = len(solutions)
    if num_solutions == 0:
        return

    for s in solutions:
        s.crowding_distance = 0

    for i in range(len(solutions[0].score)):
        solutions.sort(key=lambda s: s.score[i])
        solutions[0].crowding_distance = float('inf')
        solutions[-1].crowding_distance = float('inf')

        max_score = solutions[-1].score[i]
        min_score = solutions[0].score[i]
        if max_score == min_score:
            continue  # Skip this objective if all scores are the same

        for j in range(1, num_solutions - 1):
            if solutions[j + 1].score[i] != solutions[j - 1].score[i]:
                solutions[j].crowding_distance += (solutions[j + 1].score[i] - solutions[j - 1].score[i]) / (max_score - min_score)
            else:
                solutions[j].crowding_distance += 0