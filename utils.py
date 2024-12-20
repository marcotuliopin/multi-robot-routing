import numpy as np


def interpolate_paths(path1: list, path2: list, positions: np.ndarray, step: float) -> tuple:
    interpolated_path1 = interpolate_path(path1, positions, step)
    interpolated_path2 = interpolate_path(path2, positions, step)

    if len(interpolated_path1) > len(interpolated_path2):
        interpolated_path2 = np.vstack((interpolated_path2, [interpolated_path2[-1]] * (len(interpolated_path1) - len(interpolated_path2))))
    elif len(interpolated_path2) > len(interpolated_path1):
        interpolated_path1 = np.vstack((interpolated_path1, [interpolated_path1[-1]] * (len(interpolated_path2) - len(interpolated_path1))))

    return [interpolated_path1, interpolated_path2]


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