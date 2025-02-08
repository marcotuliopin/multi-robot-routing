import numpy as np
from .entities import Solution


def evaluate(
    solution: Solution,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> tuple[float, float]:
    paths = solution.get_solution_paths()

    max_reward = maximize_reward(paths, rvalues) / rvalues.sum() * 100

    interesting_times = get_time_to_rewards(paths, Solution.speeds, distmx)
    interpolated_positions = interpolate_positions(paths, Solution.speeds, interesting_times, rpositions, distmx)

    max_distance = calculate_max_distance(interpolated_positions)

    max_len = get_paths_max_length(paths, distmx)

    return max_reward, -max_distance, -max_len


def get_paths_max_length(paths: list[np.ndarray], distmx: np.ndarray) -> float:
    mean_distance = []
    for path in paths:
        distances = np.sum(distmx[path[:-1], path[1:]])
        mean_distance.append(distances)
    return np.max(mean_distance)


def interpolate_positions(
    paths: list[np.ndarray],
    speeds: list[float],
    interesting_times: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> np.ndarray:
    interpolated_positions = []

    for path, speed in zip(paths, speeds):
        time_to_rewards = np.cumsum(distmx[path[:-1], path[1:]]) / speed
        x_positions = rpositions[path[1:], 0]
        y_positions = rpositions[path[1:], 1]
        interpolated_x = np.interp(interesting_times, time_to_rewards, x_positions)
        interpolated_y = np.interp(interesting_times, time_to_rewards, y_positions)
        interpolated_positions.append(np.vstack((interpolated_x, interpolated_y)).T)

    return np.array(interpolated_positions)


def get_time_to_rewards(
    paths: list[np.ndarray], speeds: list[float], distmx: np.ndarray
) -> np.ndarray:
    def calculate_times(path, speed):
        return np.cumsum(distmx[path[:-1], path[1:]]) / speed

    times_to_rewards = [calculate_times(path, speed) for path, speed in zip(paths, speeds)]
    times_to_rewards = np.concatenate(times_to_rewards)

    return np.unique(times_to_rewards)


def calculate_max_distance(interpolated_positions: np.ndarray) -> float:
    max_distance = 0.0
    num_paths = len(interpolated_positions)

    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            distances = np.linalg.norm(
                interpolated_positions[i] - interpolated_positions[j], axis=1
            )
            max_distance = max(max_distance, np.max(distances))

    return max_distance


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


def update_archive(
    archive: list[Solution], neighbors: list[Solution], archive_max_size: int
) -> tuple:
    all_solutions = archive + neighbors

    non_dominated, dominated = get_non_dominated_solutions(all_solutions)

    if len(non_dominated) > archive_max_size:
        non_dominated = select_by_crowding_distance(non_dominated, archive_max_size)

    # If there is space left in the archive, add the non-dominated solutions
    selected_dominated = []
    if len(non_dominated) < archive_max_size:
        selected_dominated = select_by_crowding_distance(
            dominated, min(archive_max_size - len(non_dominated), len(dominated))
        )

    archive = (
        non_dominated
        + selected_dominated[: max(0, archive_max_size - len(non_dominated))]
    )

    return archive, non_dominated, selected_dominated


def get_non_dominated_solutions(
    solutions: list[Solution],
) -> tuple[list[Solution], list[Solution]]:
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
            if not any(
                right_non_dominated[k].dominates(left_non_dominated[i])
                for k in range(j, len(right_non_dominated))
            ):
                non_dominated.append(left_non_dominated[i])
            else:
                dominated.append(left_non_dominated[i])
            i += 1
        else:
            if not any(
                left_non_dominated[k].dominates(right_non_dominated[j])
                for k in range(i, len(left_non_dominated))
            ):
                non_dominated.append(right_non_dominated[j])
            else:
                dominated.append(right_non_dominated[j])
            j += 1

    non_dominated.extend(left_non_dominated[i:])
    non_dominated.extend(right_non_dominated[j:])

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
        solutions[0].crowding_distance = float("inf")
        solutions[-1].crowding_distance = float("inf")

        max_score = solutions[-1].score[i]
        min_score = solutions[0].score[i]
        if max_score == min_score:
            continue  # Skip this objective if all scores are the same

        for j in range(1, num_solutions - 1):
            if solutions[j + 1].score[i] != solutions[j - 1].score[i]:
                solutions[j].crowding_distance += (
                    solutions[j + 1].score[i] - solutions[j - 1].score[i]
                ) / (max_score - min_score)
            else:
                solutions[j].crowding_distance += 0
