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
    max_reward = maximize_reward(bounded_paths[0], bounded_paths[1], rvalues) + maximize_reward(bounded_paths[1], bounded_paths[0], rvalues)
    max_communication = calculate_rssi(bounded_paths[0], bounded_paths[1], rpositions)
    return max_reward, max_communication


def maximize_reward(path1: np.ndarray, path2, rvalues: np.ndarray) -> float:
    diff_elements = path1[~np.isin(path1, path2)]
    common_elements = path1[np.isin(path1, path2)]

    return rvalues[diff_elements].sum() + rvalues[common_elements].sum() / 2


def update_archive(archive: list[Solution], neighbors: list[Solution], archive_max_size: int) -> tuple:
    all_solutions = archive + (neighbors)

    non_dominated = []
    dominated = []
    assigned = set()
    for i in range(len(all_solutions)):
        for j in range(i + 1, len(all_solutions)):
            if i in assigned or j in assigned:
                continue

            if all_solutions[i].dominates(all_solutions[j]):
                dominated.append(all_solutions[j])
                assigned.add(j)
            elif all_solutions[j].dominates(all_solutions[i]):
                dominated.append(all_solutions[i])
                assigned.add(i)

        if i not in assigned:
            non_dominated.append(all_solutions[i])
    
    if len(non_dominated) > archive_max_size:
        non_dominated = select_by_crowding_distance(non_dominated, archive_max_size)
        print("Crowding distance", len(non_dominated))

    # If there is space left in the archive, add the non-dominated solutions
    selected_dominated = []
    if len(non_dominated) < archive_max_size:
        selected_dominated = select_by_crowding_distance(dominated, min(archive_max_size - len(non_dominated), len(dominated)))

    return non_dominated + selected_dominated, non_dominated, selected_dominated


def select_by_crowding_distance(archive: list[Solution], k: int) -> list[Solution]:
    assign_crowding_distance(archive)
    archive.sort(key=lambda s: s.crowding_distance, reverse=True)
    return archive[:k]


def assign_crowding_distance(archive: list[Solution]) -> None:
    num_solutions = len(archive)
    if num_solutions == 0:
        return

    for s in archive:
        s.crowding_distance = 0

    for i in range(len(archive[0].score)):
        archive.sort(key=lambda s: s.score[i])
        archive[0].crowding_distance = float('inf')
        archive[-1].crowding_distance = float('inf')

        for j in range(1, num_solutions - 1):
            archive[j].crowding_distance += archive[j + 1].score[i] - archive[j - 1].score[i]
