import numpy as np

def interpolate_paths(paths: list, positions: np.ndarray, step: float) -> list:
    interpolated_paths = [interpolate_path(path, positions, step) for path in paths]
    
    max_length = max(len(p) for p in interpolated_paths)
    for i in range(len(interpolated_paths)):
        if len(interpolated_paths[i]) < max_length:
            interpolated_paths[i] = np.vstack((interpolated_paths[i], [interpolated_paths[i][-1]] * (max_length - len(interpolated_paths[i]))))
    
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
        idx = np.searchsorted(cum_dist, d, side='right') - 1
        idx = min(idx, len(path) - 2)  # Avoid index out of bounds
        t = (d - cum_dist[idx]) / (cum_dist[idx + 1] - cum_dist[idx])  # Calculate interpolation factor

        # Interpolate between the points
        start, end = positions[path[idx]], positions[path[idx + 1]]
        sampled_points.append((1 - t) * start + t * end)

    return np.array(sampled_points)

def translate_path_to_coordinates(path: list, positions: np.ndarray) -> list:
    return [positions[reward] for reward in path]

def calculate_rssi(
    paths: list,
    rpositions: np.ndarray,
    tx_power: float = -30,
    path_loss_exponent: float = 2.0,
    noise_std: float = 1.0,
) -> float:
    interpolated_points = interpolate_paths(paths, rpositions, 1)
    distances = [np.linalg.norm(interpolated_points[i] - interpolated_points[j], axis=1) for i in range(len(paths)) for j in range(i + 1, len(paths))]
    max_distance = np.max(distances)

    if max_distance < 1e-3:
        max_distance = 0.1

    rssi = tx_power - 10 * path_loss_exponent * np.log10(max_distance)
    rssi += np.random.normal(0, noise_std)

    return rssi

def calculate_rssi_history(
    path1: np.ndarray,
    path2: np.ndarray,
    rpositions: np.ndarray,
    tx_power: float = -30,
    path_loss_exponent: float = 2.0,
    noise_std: float = 1.0,
    step: float = 1.0,
) -> np.ndarray:
    interpolated_points = interpolate_paths([path1, path2], rpositions, step)
    interpolated_points = np.hstack((interpolated_points, np.zeros((len(interpolated_points), 1, 2))))
    distances = np.linalg.norm(interpolated_points[0] - interpolated_points[1], axis=1)
    rssi_history = []
    for distance in distances:
        if distance < 1e-3:
            distance = 0.1

        rssi = tx_power - 10 * path_loss_exponent * np.log10(distance)
        rssi += np.random.normal(0, noise_std)
        rssi_history.append(rssi)
    return rssi_history