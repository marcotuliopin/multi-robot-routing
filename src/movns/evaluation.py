import numpy as np
from .entities import Solution
from utils import calculate_rssi

# Time complexity: O(n^2 * m), where n is the number of paths and m is the number of points in each path.
# Space complexity: O(1), as it uses a constant amount of additional space.
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

# Time complexity: O(n), where n is the number of points in the paths.
# Space complexity: O(1), as it uses a constant amount of additional space.
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

# Time complexity: O(n log n), where n is the number of solutions in the archive and neighbors.
# Space complexity: O(n), as it stores the non-dominated and dominated solutions.
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

# Time complexity: O(n log n), where n is the number of solutions.
# Space complexity: O(n), as it stores the non-dominated and dominated solutions.
def get_non_dominated_solutions(solutions: list[Solution]) -> tuple[list[Solution], list[Solution]]:
    if len(solutions) <= 1:
        return solutions, []

    mid = len(solutions) // 2
    left_non_dominated, left_dominated = get_non_dominated_solutions(solutions[:mid])
    right_non_dominated, right_dominated = get_non_dominated_solutions(solutions[mid:])

    non_dominated = []
    dominated = left_dominated + right_dominated

    i = j = 0
    while i < len(left_non_dominated) and j < len(right_non_dominated):
        if left_non_dominated[i].score <= right_non_dominated[j].score:
            if not any(right_non_dominated[k].dominates(left_non_dominated[i]) for k in range(j, len(right_non_dominated))):
                non_dominated.append(left_non_dominated[i])
            else:
                dominated.append(left_non_dominated[i])
            i += 1
        else:
            if not any(left_non_dominated[k].dominates(right_non_dominated[j]) for k in range(i, len(left_non_dominated))):
                non_dominated.append(right_non_dominated[j])
            else:
                dominated.append(right_non_dominated[j])
            j += 1

    non_dominated.extend(left_non_dominated[i:])
    non_dominated.extend(right_non_dominated[j:])

    return non_dominated, dominated

# Time complexity: O(n log n), where n is the number of solutions.
# Space complexity: O(n), as it stores the selected solutions.
def select_by_crowding_distance(solutions: list[Solution], k: int) -> list[Solution]:
    assign_crowding_distance(solutions)
    solutions.sort(key=lambda s: s.crowding_distance, reverse=True)
    return solutions[:k]

# Time complexity: O(n log n), where n is the number of solutions.
# Space complexity: O(1), as it uses a constant amount of additional space.
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