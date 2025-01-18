import numpy as np
from heapq import merge
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

    # Calculate the RSSI in the segments that are performed at the same time
    travel_times = solution.get_travel_times(bounded_paths, distmx)
    concurrent_segments = find_overlapping_segments_in_time(travel_times)
    interest_points = get_concurrent_segment_points(
        concurrent_segments, bounded_paths, travel_times, rpositions, distmx
    )
    max_communication = -np.inf
    for point1, point2 in interest_points:
        distance = np.linalg.norm(point1 - point2, axis=1)
        rssi = calculate_rssi(distance)
        max_communication = max(max_communication, rssi)

    return max_reward, max_communication


def maximize_reward(paths: list[np.ndarray], rvalues: np.ndarray) -> float:
    all_elements = np.concatenate(paths)
    unique_elements = np.unique(all_elements)
    reward = sum(rvalues[element] for element in unique_elements)
    return reward


def find_overlapping_segments_in_time(
    travel_times: list[list[float]],
) -> list[list[int]]:
    all_segments = merge(*travel_times)
    concurrent_segments = []
    open_segments = set()

    for segment in all_segments:
        if segment[1] == "begin":
            # Saves the agent and number of the segment
            new_open_segment = (
                segment[2],
                segment[3],
            )
            # If the segment is already open, it means that the two segments being performed at the same time
            for open_segment in open_segments:
                concurrent_segments.append((open_segment, new_open_segment))
            open_segments.add(new_open_segment)
        else:
            open_segments.remove((segment[2], segment[3]))

    return concurrent_segments


def get_concurrent_segment_points(
    concurrent_segments: list[list[int]],
    bounded_paths: list[np.ndarray],
    travel_times: list[list[tuple]],
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> list[int]:
    interest_points = []
    for concurrent_segment in concurrent_segments:
        agent1, segment1 = concurrent_segment[0]
        agent2, segment2 = concurrent_segment[1]

        start_time1 = travel_times[agent1][segment1][0]
        start_time2 = travel_times[agent2][segment2][0]
        end_time1 = travel_times[agent1][segment1 + 1][0]
        end_time2 = travel_times[agent2][segment2 + 1][0]

        segment_start_time = max(start_time1, start_time2)
        segment_end_time = min(end_time1, end_time2)

        total_time1 = end_time1 - start_time1
        p_start1 = interpolate_point(
            rpositions[bounded_paths[agent1][segment1]],
            rpositions[bounded_paths[agent1][segment1 + 1]],
            segment_start_time,
            total_time1,
            distmx[
                bounded_paths[agent1][segment1], bounded_paths[agent1][segment1 + 1]
            ],
            Solution._SPEED,
        )

        total_time2 = end_time2 - start_time2
        p_start2 = interpolate_point(
            rpositions[bounded_paths[agent2][segment2]],
            rpositions[bounded_paths[agent2][segment2 + 1]],
            segment_start_time,
            total_time2,
            distmx[
                bounded_paths[agent1][segment1], bounded_paths[agent1][segment1 + 1]
            ],
            Solution._SPEED,
        )

        interest_points.append((p_start1, p_start2))

        p_end1 = interpolate_point(
            rpositions[bounded_paths[agent1][segment1]],
            rpositions[bounded_paths[agent1][segment1 + 1]],
            segment_end_time,
            total_time1,
            distmx[
                bounded_paths[agent1][segment1], bounded_paths[agent1][segment1 + 1]
            ],
            Solution._SPEED,
        )

        p_end2 = interpolate_point(
            rpositions[bounded_paths[agent2][segment2]],
            rpositions[bounded_paths[agent2][segment2 + 1]],
            segment_end_time,
            total_time2,
            distmx[
                bounded_paths[agent2][segment2], bounded_paths[agent2][segment2 + 1]
            ],
            Solution._SPEED,
        )

        interest_points.append((p_end1, p_end2))

    return interest_points


def interpolate_point(
    start_pos: int,
    end_pos: int,
    time: float,
    total_time: float,
    distance: float,
    speed: float,
) -> np.ndarray:
    elapsed_distance = speed * (time / total_time) * distance

    direction = end_pos - start_pos
    direction /= np.linalg.norm(direction)

    return start_pos + elapsed_distance * direction


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
