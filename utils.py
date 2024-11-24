import numpy as np
from sklearn.neighbors import KDTree
from deap import tools


def get_last_valid_idx(path: list, distmx: np.ndarray, budget: int) -> int:
    """
    Get the last valid index of a path given a budget constraint.
    
    Parameters
    ----------
    path : list
        A list of indices representing the path.
    distmx : np.ndarray
        A distance matrix.
    budget : int
        The budget constraint.
        
    Returns
    -------
    int
        The last valid index.
    """
    total_distance = 0
    curr_reward = 0

    valid_idx = 0
    for i in range(len(path)):
        next_reward = path[i]
        total_distance += distmx[curr_reward, next_reward] + distmx[next_reward, 0]
        if total_distance > budget:
            break
        valid_idx = i
        curr_reward = next_reward

    return valid_idx


def calculate_mutation_probability(generation: int, max_generations: int, initial_prob: float, decay_rate: float) -> float:
    return initial_prob * np.exp(-decay_rate * (generation / max_generations))


def get_points_in_range(p: int, rpositions: np.ndarray, maxdist: float, kdtree: KDTree) -> list:
    current_point = rpositions[p]
    indices_within_radius = kdtree.query_radius([current_point], r=maxdist)[0]
    return indices_within_radius

    
def disentangle_paths(path1: np.ndarray, path2: np.ndarray, positions: np.ndarray) -> tuple:
    """
    Disentangle two paths that intersect. The paths are disentangled by swapping the last intersecting point.
    
    Parameters
    ----------
    path1 : np.ndarray
        The first path.
    path2 : np.ndarray
        The second path.
    positions : np.ndarray
        The positions of the rewards.
        
    Returns
    -------
    tuple
        The disentangled paths.
    """
    def segments_intersect(p1, p2, q1, q2):
        """
        Check if two line segments intersect.
        """
        def ccw(a, b, c):
            """
            Check if three points are in counter-clockwise order.
            """
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    path1, path2 = path1.copy(), path2.copy()

    for i in range(min(len(path1), len(path2)) - 1):
        p1, p2 = positions[path1[i]], positions[path1[i + 1]]
        q1, q2 = positions[path2[i]], positions[path2[i + 1]]

        if segments_intersect(p1, p2, q1, q2):
            path1[i + 1], path2[i + 1] = path2[i + 1], path1[i + 1]

    return path1, path2


def assign_crowding_dist(population):
    tools.sortNondominated(population, len(population), first_front_only=False)

    for front in population:
        front_size = len(front)
        if front_size > 0:
            front[0].fitness.crowding_dist = float('inf')
            front[-1].fitness.crowding_dist = float('inf')

            for i in range(len(front[0].fitness.values)):
                crowd = sorted([(ind.fitness.values, ind) for ind in front], key=lambda element: element[0][i])
                
                if crowd[-1][0][i] == crowd[0][0][i]:
                    continue
                
                norm = float(crowd[-1][0][i] - crowd[0][0][i])
                
                for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
                    dist = (next[0][i] - prev[0][i]) / norm
                    cur[1].fitness.crowding_dist += dist


def interpolate_paths(path1: list, path2: list, positions: np.ndarray, num_samples: int = 100) -> tuple:
    """
    Interpolate two paths such that the samples are equally spaced along both paths.

    Parameters
    ----------
    path1 : list
        The first path.
    path2 : list
        The second path.
    positions : np.ndarray
        The positions of the rewards.
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    tuple
        Two arrays of interpolated points for path1 and path2.
    """
    def cumulative_distances(path):
        distances = [0.0]
        for i in range(len(path) - 1):
            start, end = positions[path[i]], positions[path[i + 1]]
            distances.append(distances[-1] + np.linalg.norm(end - start))
        return np.array(distances)

    # Interpolate points for both paths
    def interpolate_path(cum_dist, path):
        interpolated_points = []
        for d in sample_distances:
            idx = np.searchsorted(cum_dist, d, side='right') - 1
            idx = min(idx, len(path) - 2)  # Avoid index out of bounds
            t = (d - cum_dist[idx]) / (cum_dist[idx + 1] - cum_dist[idx])
            start, end = positions[path[idx]], positions[path[idx + 1]]
            interpolated_points.append((1 - t) * start + t * end)
        return np.array(interpolated_points)
    
    # Calculate cumulative distances for both paths
    cum_dist1 = cumulative_distances(path1)
    cum_dist2 = cumulative_distances(path2)

    # Define the sampling points (shared between the two paths)
    total_distance = max(cum_dist1[-1], cum_dist2[-1])
    sample_distances = np.linspace(0, total_distance, num_samples)
    
    interpolated_path1 = interpolate_path(cum_dist1, path1)
    interpolated_path2 = interpolate_path(cum_dist2, path2)
    
    return interpolated_path1, interpolated_path2


def translate_path_to_coordinates(path: list, positions: np.ndarray) -> list:
    """
    Translate a path of indices to a path of coordinates.

    Parameters
    ----------
    path : list
        The path of indices.
    positions : np.ndarray
        The positions of the rewards.

    Returns
    -------
    list
        The path of coordinates.
    """
    return [positions[reward] for reward in path]
