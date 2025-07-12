import numpy as np
from numba import njit

from utils import calculate_rssi
from .entities import Solution


def evaluate(
    solution: Solution,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> tuple[float, float]:
    paths = solution.get_solution_paths()
    
    # Convert to formats compatible with Numba functions
    speeds = np.array(Solution.speeds)
    paths_flat = np.concatenate(paths)

    max_reward = maximize_reward(paths_flat, rvalues)

    interesting_times = get_time_to_rewards(paths, speeds, distmx)
    interpolated_positions = interpolate_positions(paths, speeds, interesting_times, rpositions, distmx)

    max_distance = calculate_max_distance(interpolated_positions)
    max_rssi = calculate_rssi(max_distance, noise=False)

    min_len = get_paths_max_length(paths, distmx)

    # We are trying to maximize all these. Since rssi is a negative measure, maximizing it means enhancing connectivity.
    return max_reward, max_rssi, -min_len


@njit(cache=True, fastmath=True)
def get_paths_max_length(paths_array: np.ndarray, distmx: np.ndarray) -> float:
    max_distance = 0.0
    
    for i in range(len(paths_array)):
        path = paths_array[i]
        distances = 0.0
        for j in range(len(path) - 1):
            distances += distmx[path[j], path[j + 1]]
        if distances > max_distance:
            max_distance = distances
            
    return max_distance


def get_paths_max_length(paths: list[np.ndarray], distmx: np.ndarray) -> float:
    max_distance = 0.0
    
    for path in paths:
        distances = 0.0
        for j in range(len(path) - 1):
            distances += distmx[path[j], path[j + 1]]
        max_distance = max(max_distance, distances)
            
    return max_distance


def interpolate_positions(
    paths: list[np.ndarray],
    speeds: np.ndarray,
    interesting_times: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
) -> np.ndarray:
    num_paths = len(paths)
    num_times = len(interesting_times)
    interpolated_positions = np.zeros((num_paths, num_times, 2))
    
    for i in range(num_paths):
        path = paths[i]
        speed = speeds[i]

        if len(path) <= 1:
            continue
            
        time_to_rewards = np.zeros(len(path) - 1)
        cumulative_dist = 0.0
        for j in range(len(path) - 1):
            cumulative_dist += distmx[path[j], path[j + 1]]
            time_to_rewards[j] = cumulative_dist / speed
        
        x_positions = np.zeros(len(path) - 1)
        y_positions = np.zeros(len(path) - 1)
        for j in range(len(path) - 1):
            x_positions[j] = rpositions[path[j + 1], 0]
            y_positions[j] = rpositions[path[j + 1], 1]
        
        # Interpolate for each interesting time
        for t in range(num_times):
            if len(time_to_rewards) > 0:
                interpolated_positions[i, t, 0] = np.interp(interesting_times[t], time_to_rewards, x_positions)
                interpolated_positions[i, t, 1] = np.interp(interesting_times[t], time_to_rewards, y_positions)
    
    return interpolated_positions


@njit(cache=True, fastmath=True)
def get_time_to_rewards(
    paths: list[np.ndarray], speeds: np.ndarray, distmx: np.ndarray
) -> np.ndarray:
    all_times = []
    
    for path, speed in zip(paths, speeds):
        cumulative_dist = 0.0
        for j in range(len(path) - 1):
            cumulative_dist += distmx[path[j], path[j + 1]]
            all_times.append(cumulative_dist / speed)
    
    if len(all_times) == 0:
        return np.array([0.0])
    
    times_array = np.array(all_times)
    return np.unique(times_array)


@njit(cache=True, fastmath=True)
def calculate_max_distance(interpolated_positions: np.ndarray) -> float:
    if len(interpolated_positions) <= 1:
        return 0.0
    
    k, n, _ = interpolated_positions.shape
    max_distance = 0.0
    
    for i in range(k):
        for j in range(i + 1, k):
            for t in range(n):
                pos_i = interpolated_positions[i, t, :]
                pos_j = interpolated_positions[j, t, :]
                
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)

                max_distance = max(max_distance, distance) 
    
    return max_distance


@njit(cache=True, fastmath=True)
def maximize_reward(paths_flat: np.ndarray, rvalues: np.ndarray) -> float:
    unique_elements = np.unique(paths_flat)
    reward = 0.0
    for element in unique_elements:
        reward += rvalues[element]
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
    
    # Use NSGA-II style fast non-dominated sorting
    return fast_non_dominated_sort(solutions)


def fast_non_dominated_sort(solutions: list[Solution]) -> tuple[list[Solution], list[Solution]]:
    n = len(solutions)
    if n <= 1:
        return solutions, []
    
    # Initialize dominance structures
    domination_count = [0] * n  # Number of solutions that dominate solution i
    dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by solution i
    
    # Calculate dominance relationships - O(N²M)
    for i in range(n):
        for j in range(i + 1, n):
            if solutions[i].dominates(solutions[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif solutions[j].dominates(solutions[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1
    
    # Find non-dominated solutions (domination_count = 0)
    non_dominated = []
    dominated = []
    
    for i in range(n):
        if domination_count[i] == 0:
            non_dominated.append(solutions[i])
        else:
            dominated.append(solutions[i])
    
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
