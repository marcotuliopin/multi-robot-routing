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


def update_archive(archive: list[Solution], neighbors: list[Solution], archive_max_size: int) -> None:
    archive.extend(neighbors)

    non_dominated = []
    for s in archive:
        if not any(other.dominates(s) for other in archive if other != s):
            non_dominated.append(s)
    archive[:] = non_dominated

    if len(archive) > archive_max_size:
        archive[:] = select_by_crowding_distance(archive, archive_max_size)

    return archive


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
