import numpy as np
from numba import njit


def interpolate_paths_with_speeds(paths, speeds, rpositions, step):
    interpolated_paths = []
    max_length = 0

    for path, speed in zip(paths, speeds):
        distances = np.cumsum(np.linalg.norm(np.diff(rpositions[path], axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        total_distance = distances[-1]
        num_steps = int(total_distance * step / speed)
        print(num_steps)
        new_distances = np.linspace(0, total_distance, num_steps)
        new_path = np.zeros((num_steps, 2))
        new_path[:, 0] = np.interp(new_distances, distances, rpositions[path, 0])
        new_path[:, 1] = np.interp(new_distances, distances, rpositions[path, 1])
        interpolated_paths.append(new_path)
        if num_steps > max_length:
            max_length = num_steps

        for i in range(len(interpolated_paths)):
            if len(interpolated_paths[i]) < max_length:
                padding = np.tile(interpolated_paths[i][-1], (max_length - len(interpolated_paths[i]), 1))
                interpolated_paths[i] = np.vstack([interpolated_paths[i], padding])

    return np.array(interpolated_paths)


def interpolate_paths(paths: list, positions: np.ndarray, step: float) -> list:
    interpolated_paths = [interpolate_path(path, positions, step) for path in paths]

    max_length = max(len(p) for p in interpolated_paths)
    for i in range(len(interpolated_paths)):
        if len(interpolated_paths[i]) < max_length:
            interpolated_paths[i] = np.vstack(
                (
                    interpolated_paths[i],
                    [interpolated_paths[i][-1]]
                    * (max_length - len(interpolated_paths[i])),
                )
            )

    return interpolated_paths


def interpolate_path(path: list, positions: np.ndarray, step: float) -> np.ndarray:

    def cumulative_distances(path):
        distances = [0.0]
        for i in range(len(path) - 1):
            start, end = positions[path[i]], positions[path[i + 1]]
            distances.append(distances[-1] + np.linalg.norm(end - start))
        return np.array(distances)

    cum_dist = cumulative_distances(path)

    total_distance = cum_dist[-1]
    sample_distances = np.arange(0, total_distance, step)

    sampled_points = [positions[path[0]]]

    for d in sample_distances:
        # Find the two points between which the sample will be interpolated
        idx = np.searchsorted(cum_dist, d, side="right") - 1
        idx = min(idx, len(path) - 2)  # Avoid index out of bounds
        t = (d - cum_dist[idx]) / (
            cum_dist[idx + 1] - cum_dist[idx]
        )  # Calculate interpolation factor

        # Interpolate between the points
        start, end = positions[path[idx]], positions[path[idx + 1]]
        sampled_points.append((1 - t) * start + t * end)

    return np.array(sampled_points)


def translate_path_to_coordinates(path: list, positions: np.ndarray) -> list:
    return [positions[reward] for reward in path]


def calculate_rssi(
    distance: float,
    tx_power: float = -30,
    path_loss_exponent: float = 2.0,
    noise: bool = True,
    noise_std: float = 1.0,
) -> float:
    rssi = _calculate_rssi(distance, tx_power, path_loss_exponent)
    
    if noise:
        rssi += np.random.normal(0, noise_std)
    
    return rssi


@njit(cache=True, fastmath=True)
def _calculate_rssi(distance, tx_power, path_loss_exponent):
    if distance < 0.1:
        distance = 0.1
    return tx_power - 10 * path_loss_exponent * np.log10(distance)


def calculate_rssi_history(
    paths: list[np.ndarray],
    speeds: list[float],
    rpositions: np.ndarray,
    tx_power: float = -30,
    path_loss_exponent: float = 2.0,
    step: float = 1.0,
) -> np.ndarray:
    interpolated_points = interpolate_paths_with_speeds(paths, speeds, rpositions, step)
    interpolated_points = np.hstack((interpolated_points, np.zeros((len(interpolated_points), 1, 2))))
    num_points = interpolated_points.shape[1]

    max_distances = np.zeros(num_points)
    for i in range(num_points):
        distances = np.linalg.norm(interpolated_points[:, np.newaxis, i, :] - interpolated_points[np.newaxis, :, i, :], axis=2)
        max_distances[i] = np.max(distances)

    rssi_history = []
    for distance in max_distances:
        if distance < 1e-3:
            distance = 0.1

        rssi = tx_power - 10 * path_loss_exponent * np.log10(distance)
        # rssi += np.random.normal(0, noise_std)
        rssi_history.append(rssi)
    return rssi_history
