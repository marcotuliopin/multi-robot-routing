from numba import njit
import numpy as np

from . import Solution
from .SanityAssertions import SanityAssertions

INVERSION_PROBABILITY = 0.75
EPSILON = 1e-4

class Neighborhood:
    """
    A class that implements various neighborhood operators for local search and perturbation
    in multi-agent routing optimization problems.
    """
    
    # Constants
    
    def __init__(self, algorithm: str = "unique_vis"):
        """
        Initialize the neighborhood operators based on the specified algorithm.
        
        Args:
            algorithm: The algorithm type ("unique_vis" or other)
        """
        self._setup_operators(algorithm)
        self.num_neighborhoods = len(self.local_search_operators) * len(self.perturbation_operators)
        self.epsilon = EPSILON
    
    def _setup_operators(self, algorithm: str) -> None:
        """Setup perturbation and local search operators based on algorithm type."""
        if algorithm == "unique_vis":
            self.perturbation_operators = [
                self.invert_points_all_agents_unique,
                self.two_opt_all_paths,
            ]
            self.local_search_operators = [
                self.move_point,
                self.swap_points,
                self.invert_single_point_unique,
                self.add_and_move_unique,
                self.two_opt,
            ]
        else:
            self.perturbation_operators = [
                self.invert_points_all_agents,
                self.two_opt_all_paths,
            ]
            self.local_search_operators = [
                self.move_point,
                self.swap_points,
                self.invert_single_point,
                self.add_and_move,
                self.two_opt,
                self.path_relinking,
            ]
    
    # ==================== OPERATOR SELECTION METHODS ====================
    
    def get_perturbation_operator(self):
        """Randomly select a perturbation operator."""
        operator_index = np.random.randint(0, len(self.perturbation_operators))
        return self.perturbation_operators[operator_index]

    def get_local_search_operator(self, neighborhood: int):
        """Get a local search operator based on neighborhood index."""
        return self.local_search_operators[neighborhood % len(self.local_search_operators)]

    # ==================== PERTURBATION OPERATORS ====================

    def two_opt_all_paths(self, solution: Solution) -> Solution:
        """Apply 2-opt operation to all agent paths simultaneously."""
        new_solution = solution.copy()
        new_solution.paths = two_opt_all_paths_core(solution.paths)
        
        # Sanity check: Ensure uniqueness constraint is maintained
        SanityAssertions.assert_one_agent_per_reward(new_solution.paths)
        
        return new_solution

    def invert_points_all_agents(self, solution: Solution) -> Solution:
        """Invert points across all agents independently."""
        new_solution = solution.copy()
        new_solution.paths = invert_points_all_agents_core(solution.paths)
        return new_solution
    
    def invert_points_all_agents_unique(self, solution: Solution) -> Solution:
        """Invert points across all agents while maintaining uniqueness constraint."""
        new_solution = solution.copy()
        new_solution.paths = invert_points_all_agents_unique_core(solution.paths)
        
        # Sanity check: Ensure uniqueness constraint is maintained
        SanityAssertions.assert_one_agent_per_reward(new_solution.paths)
        
        return new_solution
    
    def untangle_path(self, solution: Solution, rpositions: np.ndarray) -> Solution:
        """
        Untangle intersecting path segments by reversing crossing segments.
        
        Args:
            solution: Current solution
            rpositions: Array of node positions for intersection detection
            
        Returns:
            Solution with untangled paths
        """
        new_solution = solution.copy()
        
        for path in new_solution.paths:
            self._untangle_single_agent_path(path, rpositions)
            
        return new_solution
    
    def _untangle_single_agent_path(self, path: np.ndarray, rpositions: np.ndarray) -> None:
        """
        Untangle intersections in a single agent's path.
        
        Args:
            path: Agent's path array (modified in-place)
            rpositions: Array of node positions
        """
        positive_indices = self._get_positive_indices(path)
        if len(positive_indices) < 4:  # Need at least 4 points to have intersections
            return
            
        sorted_indices = positive_indices[np.argsort(path[positive_indices])]
        positions = rpositions[sorted_indices]
        
        # Check all pairs of consecutive segments for intersections
        for i in range(len(sorted_indices) - 1):
            for j in range(i + 2, len(sorted_indices) - 1):  # Skip adjacent segments
                segment1_start = positions[i]
                segment1_end = positions[i + 1]
                segment2_start = positions[j]
                segment2_end = positions[j + 1]
                
                if self._segments_intersect(segment1_start, segment1_end, segment2_start, segment2_end):
                    self._reverse_path_segment(path, sorted_indices, i + 1, j)
    
    def _segments_intersect(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        """
        Check if two line segments intersect using the CCW algorithm.
        
        Args:
            a, b: Endpoints of first segment
            c, d: Endpoints of second segment
            
        Returns:
            True if segments intersect, False otherwise
        """
        return (self._ccw(a, c, d) != self._ccw(b, c, d) and 
                self._ccw(a, b, c) != self._ccw(a, b, d))
    
    def _ccw(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        """
        Check if three points are in counter-clockwise order.
        
        Args:
            p1, p2, p3: Points to check
            
        Returns:
            True if points are in counter-clockwise order
        """
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def _reverse_path_segment(self, path: np.ndarray, sorted_indices: np.ndarray, 
                            start_idx: int, end_idx: int) -> None:
        """
        Reverse a segment of the path to untangle intersections.
        
        Args:
            path: Agent's path array (modified in-place)
            sorted_indices: Sorted indices of visited nodes
            start_idx: Start index of segment to reverse
            end_idx: End index of segment to reverse
        """
        segment_indices = sorted_indices[start_idx:end_idx + 1]
        path[segment_indices] = path[segment_indices][::-1]

    # ==================== LOCAL SEARCH OPERATORS ====================

    def move_point(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by moving a point to different positions in the agent's path.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        positive_indices = self._get_positive_indices(path)

        if len(positive_indices) < 2:
            return []

        neighbors = []
        
        for source_idx in range(len(positive_indices)):
            source_position = positive_indices[source_idx]
            source_value = path[source_position]
            
            for target_idx in range(len(positive_indices)):
                if source_idx == target_idx:
                    continue
                    
                neighbor = self._create_move_neighbor(solution, agent, positive_indices, 
                                                    source_idx, target_idx, source_value)
                neighbors.append(neighbor)

        return neighbors
    
    def _create_move_neighbor(self, solution: Solution, agent: int, positive_indices: np.ndarray,
                            source_idx: int, target_idx: int, source_value: float) -> Solution:
        """Helper method to create a neighbor by moving a point."""
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        
        positive_values = new_path[positive_indices].copy()
        positive_values = np.delete(positive_values, source_idx)
        
        insertion_position = target_idx if target_idx < source_idx else target_idx - 1
        positive_values = np.insert(positive_values, insertion_position, source_value)
        
        new_path[positive_indices] = positive_values
        return new_solution

    def swap_points(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by swapping pairs of points in the agent's path.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        neighbors = []
        positive_indices = self._get_positive_indices(path)

        for i in range(len(positive_indices)):
            for j in range(i + 1, len(positive_indices)):
                neighbor = self._create_swap_neighbor(solution, agent, positive_indices[i], positive_indices[j])
                neighbors.append(neighbor)

        return neighbors
    
    def _create_swap_neighbor(self, solution: Solution, agent: int, idx1: int, idx2: int) -> Solution:
        """Helper method to create a neighbor by swapping two points."""
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        return new_solution
    
    def two_opt(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors using 2-opt operation on the agent's path.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        positive_indices = self._get_positive_indices(path)

        if len(positive_indices) < 2:
            return []

        neighbors = []

        for i in range(len(positive_indices) - 1):
            for j in range(i + 1, len(positive_indices)):
                neighbor = self._create_two_opt_neighbor(solution, agent, positive_indices, i, j)
                neighbors.append(neighbor)

        return neighbors
    
    def _create_two_opt_neighbor(self, solution: Solution, agent: int, positive_indices: np.ndarray, 
                               start_idx: int, end_idx: int) -> Solution:
        """Helper method to create a neighbor using 2-opt reversal."""
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        
        indices_to_reverse = positive_indices[start_idx:end_idx + 1]
        new_path[indices_to_reverse] = new_path[indices_to_reverse][::-1].copy()
        
        return new_solution

    def invert_single_point(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by inverting each point in the agent's path.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(path)):
            neighbor = self._create_single_invert_neighbor(solution, agent, i)
            neighbors.append(neighbor)

        return neighbors
    
    def _create_single_invert_neighbor(self, solution: Solution, agent: int, point_idx: int) -> Solution:
        """Helper method to create a neighbor by inverting a single point."""
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        new_path[point_idx] *= -1
        return new_solution

    def invert_single_point_unique(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by adding/removing points while maintaining uniqueness.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        return self.add_point_unique(solution, agent) + self.remove_point(solution, agent)
    
    def add_point_unique(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by adding points while maintaining uniqueness constraint.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        neighbors = []
        negative_indices = self._get_negative_indices(path)

        for point_idx in negative_indices:
            neighbor = self._create_unique_add_neighbor(solution, agent, point_idx)
            neighbors.append(neighbor)
        
        return neighbors
    
    def _create_unique_add_neighbor(self, solution: Solution, agent: int, point_idx: int) -> Solution:
        """Helper method to create a neighbor by adding a point uniquely."""
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        
        # Remove point from other agents to maintain uniqueness
        positive_agents = np.where(new_solution.paths[:, point_idx] > 0)[0]
        new_solution.paths[positive_agents, point_idx] *= -1
        
        # Add point to current agent
        new_path[point_idx] *= -1
        
        return new_solution

    def remove_point(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by removing points from the agent's path.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        neighbors = []
        positive_indices = self._get_positive_indices(path)

        for point_idx in positive_indices:
            neighbor = self._create_remove_neighbor(solution, agent, point_idx)
            neighbors.append(neighbor)

        return neighbors
    
    def _create_remove_neighbor(self, solution: Solution, agent: int, point_idx: int) -> Solution:
        """Helper method to create a neighbor by removing a point."""
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        new_path[point_idx] = -new_path[point_idx]
        return new_solution

    def add_and_move(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by adding a point and then moving it to different positions.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        neighbors = []
        negative_indices = self._get_negative_indices(path)

        for add_point_idx in negative_indices:
            neighbors.extend(self._create_add_and_move_neighbors(solution, agent, add_point_idx))

        return neighbors
    
    def _create_add_and_move_neighbors(self, solution: Solution, agent: int, add_point_idx: int) -> list[Solution]:
        """Helper method to create neighbors by adding a point and moving it."""
        # First add the point
        temp_solution = solution.copy()
        temp_path = temp_solution.paths[agent]
        temp_path[add_point_idx] *= -1

        new_positive_indices = self._get_positive_indices(temp_path)
        source_value = temp_path[add_point_idx]
        neighbors = []

        # Then move it to different positions
        for target_idx in new_positive_indices:
            if add_point_idx == target_idx:
                continue
                
            neighbor = self._create_move_after_add_neighbor(
                temp_solution, agent, new_positive_indices, add_point_idx, target_idx, source_value
            )
            neighbors.append(neighbor)
        
        return neighbors
    
    def _create_move_after_add_neighbor(self, solution: Solution, agent: int, 
                                      positive_indices: np.ndarray, add_idx: int, 
                                      target_idx: int, source_value: float) -> Solution:
        """Helper method to create a neighbor by moving a point after adding it."""
        new_solution = solution.copy()
        new_path = new_solution.paths[agent]
        
        positive_values = new_path[positive_indices].copy()
        
        source_rel_idx = np.where(positive_indices == add_idx)[0][0]
        target_rel_idx = np.where(positive_indices == target_idx)[0][0]
        
        positive_values = np.delete(positive_values, source_rel_idx)
        insert_position = target_rel_idx if target_rel_idx < source_rel_idx else target_rel_idx - 1
        positive_values = np.insert(positive_values, insert_position, source_value)
        
        new_path[positive_indices] = positive_values
        return new_solution

    def add_and_move_unique(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors by adding a point uniquely and then moving it to different positions.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        neighbors = []
        negative_indices = self._get_negative_indices(path)

        for add_point_idx in negative_indices:
            neighbors.extend(self._create_add_and_move_unique_neighbors(solution, agent, add_point_idx))

        return neighbors
    
    def _create_add_and_move_unique_neighbors(self, solution: Solution, agent: int, add_point_idx: int) -> list[Solution]:
        """Helper method to create neighbors by adding a point uniquely and moving it."""
        # First add the point uniquely
        temp_solution = solution.copy()
        temp_path = temp_solution.paths[agent]

        # Remove point from other agents to maintain uniqueness
        positive_agents = np.where(temp_solution.paths[:, add_point_idx] > 0)[0]
        temp_solution.paths[positive_agents, add_point_idx] *= -1
        temp_path[add_point_idx] *= -1

        new_positive_indices = self._get_positive_indices(temp_path)
        source_value = temp_path[add_point_idx]
        neighbors = []

        # Then move it to different positions
        for target_idx in new_positive_indices:
            if add_point_idx == target_idx:
                continue
                
            neighbor = self._create_move_after_add_neighbor(
                temp_solution, agent, new_positive_indices, add_point_idx, target_idx, source_value
            )
            neighbors.append(neighbor)
        
        return neighbors

    def path_relinking(self, solution: Solution, agent: int) -> list[Solution]:
        """
        Generate neighbors using path relinking with other agents.
        
        Args:
            solution: Current solution
            agent: Agent index to modify
            
        Returns:
            List of neighbor solutions
        """
        path = solution.paths[agent]
        neighbors = []

        for other_agent in range(len(solution.paths)):
            if other_agent == agent:
                continue

            neighbors.extend(self._create_path_relink_neighbors(solution, agent, other_agent, path))

        return neighbors
    
    def _create_path_relink_neighbors(self, solution: Solution, agent: int, 
                                    other_agent: int, path: np.ndarray) -> list[Solution]:
        """Helper method to create neighbors using path relinking."""
        neighbors = []
        num_changes = np.random.randint(1, len(path))
        indices_to_change = np.random.choice(
            range(len(path)), 
            size=num_changes,
            replace=False
        )

        for point_idx in indices_to_change:
            other_path = solution.paths[other_agent]
            if other_path[point_idx] == path[point_idx]:
                continue
                
            new_solution = solution.copy()
            new_other_path = new_solution.paths[other_agent]
            new_other_path[point_idx] = path[point_idx] + np.random.random() * self.epsilon
            neighbors.append(new_solution)

        return neighbors

    # ==================== UTILITY METHODS ====================
    
    def _get_positive_indices(self, path: np.ndarray) -> np.ndarray:
        """Get indices where path values are positive."""
        return np.where(path > 0)[0]
    
    def _get_negative_indices(self, path: np.ndarray) -> np.ndarray:
        """Get indices where path values are negative."""
        return np.where(path < 0)[0]


@njit(cache=True)
def copy_paths(paths):
    """Create a copy of the paths array."""
    return paths.copy()


@njit(cache=True)
def get_positive_indices(path):
    """Get indices where path values are positive."""
    return np.where(path > 0)[0]


@njit(cache=True)
def get_negative_indices(path):
    """Get indices where path values are negative."""
    return np.where(path < 0)[0]


@njit(cache=True, fastmath=True)
def two_opt_all_paths_core(paths):
    """
    Apply 2-opt operation to all agent paths simultaneously.
    
    Args:
        paths: Array of agent paths
        
    Returns:
        Modified paths array
    """
    new_paths = copy_paths(paths)
    
    for agent_idx in range(len(new_paths)):
        path = new_paths[agent_idx]
        
        # Get indices where values are positive (visited nodes)
        positive_indices = get_positive_indices(path)
        
        if len(positive_indices) < 2:
            continue  # Need at least 2 positive values for 2-opt
            
        # Select two random positions within the positive indices
        pos1 = np.random.randint(0, len(positive_indices))
        pos2 = np.random.randint(0, len(positive_indices))
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
        if pos1 == pos2:
            continue  # No reversal needed if same position
        
        # Extract the positive values in their current order
        positive_values = path[positive_indices].copy()
        
        # Reverse the segment between pos1 and pos2 (inclusive)
        positive_values[pos1:pos2 + 1] = positive_values[pos1:pos2 + 1][::-1]
        
        # Put the reordered positive values back in their positions
        path[positive_indices] = positive_values
    
    return new_paths


@njit(cache=True, fastmath=True)
def invert_points_all_agents_core(paths):
    """
    Invert points across all agents independently.
    
    Args:
        paths: Array of agent paths
        
    Returns:
        Modified paths array
    """
    new_paths = copy_paths(paths)
    path_length = new_paths.shape[1]
    
    for agent_idx in range(len(new_paths)):
        points_to_invert = np.random.random(path_length) < INVERSION_PROBABILITY
        new_paths[agent_idx][points_to_invert] *= -1
    
    return new_paths


@njit(cache=True, fastmath=True)
def invert_points_all_agents_unique_core(paths):
    """
    Invert points across all agents while maintaining uniqueness constraint.
    
    Args:
        paths: Array of agent paths
        
    Returns:
        Modified paths array
    """
    new_paths = copy_paths(paths)
    path_length = new_paths.shape[1]
    
    points_to_invert = np.random.random(path_length) < INVERSION_PROBABILITY
    
    for point_idx in np.where(points_to_invert)[0]:
        # Remove point from all agents that currently have it
        agents_with_point = np.where(new_paths[:, point_idx] > 0)[0]
        if len(agents_with_point) > 0:
            new_paths[agents_with_point, point_idx] *= -1
        
        # Assign point to a random agent
        new_agent = np.random.randint(0, len(new_paths))
        new_paths[new_agent, point_idx] = -new_paths[new_agent, point_idx]
    
    return new_paths
