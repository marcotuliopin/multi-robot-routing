import copy
import numpy as np

from .SanityAssertions import SanityAssertions


class Solution:
    """
    Represents a solution for multi-agent routing optimization problems.
    
    A solution consists of paths for multiple agents, where each path is represented
    as an array of values. Positive values indicate visited nodes, negative values
    indicate unvisited nodes, and the order is determined by the magnitude of values.
    """
    
    # Class-level configuration parameters
    begin: int = -1
    end: int = -2
    num_agents: int = 1
    speeds: list[float] = [1]
    budget: list[int] = [150]

    def __init__(
        self,
        distmx: np.ndarray = None,
        rvalues: np.ndarray = None,
        paths: np.ndarray = None,
        score: tuple = (-1, -1, -1),
    ) -> None:
        """
        Initialize a Solution instance.
        
        Args:
            distmx: Distance matrix between nodes
            rvalues: Reward values for each node
            paths: Pre-defined paths for agents (if None, generates random paths)
            score: Multi-objective score tuple
        """
        self.score = score
        self.crowding_distance = -1
        self.visited = False
        self.dominated = True

        if paths is None:
            num_rewards = len(rvalues) - 2  # Exclude begin and end nodes
            self.paths = self.init_paths_unique(num_rewards)
            self.paths = self.bound_all_paths(self.paths, distmx, rvalues)
        else:
            self.paths = paths
    
    # ==================== CLASS CONFIGURATION METHODS ====================
        
    @classmethod
    def set_parameters(
        cls, begin: int, end: int, num_agents: int, budget: list[int], speeds: list[float]
    ) -> None:
        """
        Set class-level parameters for all Solution instances.
        
        Args:
            begin: Index of the starting node
            end: Index of the ending node
            num_agents: Number of agents in the system
            budget: Budget constraints for each agent
            speeds: Speed values for each agent
        """
        cls.begin = begin
        cls.end = end
        cls.num_agents = num_agents
        cls.budget = budget
        cls.speeds = speeds

    # ==================== PATH INITIALIZATION METHODS ====================

    def init_paths(self, num_rewards: int) -> np.ndarray:
        """
        Initialize random paths for all agents (without uniqueness constraint).
        
        Args:
            num_rewards: Number of reward nodes to consider
            
        Returns:
            Array of initialized paths for all agents
            
        Raises:
            ValueError: If paths contain duplicate values
        """
        paths = np.random.uniform(
            low=0, high=1, size=(Solution.num_agents, num_rewards)
        )
        
        if any(len(np.unique(path)) != len(path) for path in paths):
            raise ValueError("Paths must have unique values.")

        return paths
    
    def init_paths_unique(self, num_rewards: int) -> np.ndarray:
        """
        Initialize random paths ensuring each reward is assigned to exactly one agent.
        
        Args:
            num_rewards: Number of reward nodes to consider
            
        Returns:
            Array of initialized paths with uniqueness constraint
        """
        paths = np.random.uniform(
            low=-1, high=0, size=(Solution.num_agents, num_rewards)
        )

        positive_indices = np.random.randint(0, Solution.num_agents, size=(num_rewards))

        for reward_idx, agent_idx in enumerate(positive_indices):
            paths[agent_idx, reward_idx] = -paths[agent_idx, reward_idx]

        return paths

    def dominates(self, other: "Solution") -> bool:
        """
        Check if this solution dominates another solution using Pareto dominance.
        
        A solution dominates another if it's at least as good in all objectives
        and strictly better in at least one objective.
        
        Args:
            other: Another solution to compare against
            
        Returns:
            True if this solution dominates the other, False otherwise
        """
        if all(own_score == other_score for own_score, other_score in zip(self.score, other.score)):
            return True

        # Check Pareto dominance
        is_better_in_any = False
        for own_score, other_score in zip(self.score, other.score):
            if own_score > other_score:
                is_better_in_any = True
            elif own_score < other_score:
                return False
                
        return is_better_in_any

    def get_solution_paths(self) -> list[np.ndarray]:
        """
        Extract complete trajectories for each agent including start and end nodes.
        
        Returns:
            List of complete trajectories for each agent
        """
        trajectories = [self._get_sorted_indices(path) for path in self.paths]
        solution = [
            np.concatenate(([Solution.begin], trajectory, [Solution.end]))
            for trajectory in trajectories
        ]
        return solution

    def get_solution_length(self, distmx: np.ndarray) -> np.ndarray:
        """
        Calculate the total length of each agent's path.
        
        Args:
            distmx: Distance matrix between nodes
            
        Returns:
            Array of path lengths for each agent
        """
        paths = self.get_solution_paths()
        lengths = np.sum(distmx[paths[:, :-1], paths[:, 1:]], axis=1)
        return lengths

    def bound_all_paths(
        self, paths: np.ndarray, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        """
        Apply budget constraints to all agent paths.
        
        Args:
            paths: Agent paths to constrain
            distmx: Distance matrix between nodes
            rvalues: Reward values for each node
            
        Returns:
            Budget-constrained paths
        """
        for agent_idx in range(len(paths)):
            paths[agent_idx] = self.bound_path(
                paths[agent_idx], Solution.budget[agent_idx], distmx, rvalues
            )
        return paths

    def bound_path(
        self, path: np.ndarray, budget: int, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        """
        Enforce budget constraint on a single agent's path by removing nodes.
        
        Args:
            path: Agent's path to constrain
            budget: Budget limit for this agent
            distmx: Distance matrix between nodes
            rvalues: Reward values for each node
            
        Returns:
            Budget-constrained path
        """
        positive_indices = np.where(path > 0)[0]
        trajectory = positive_indices[np.argsort(path[positive_indices])]

        if len(positive_indices) == 0:
            return path

        total_length = self._calculate_path_length(trajectory, distmx)

        while total_length > budget:
            impacts = self._calculate_removal_impacts(trajectory, distmx)

            # The chance of removing a node is proportional to the impact of removing it, 
            # but inversely proportional to the reward of the node.
            # Since the reward is always positive, the ratio is always a real number greater than zero.
            reward_impact_ratio = impacts / rvalues[trajectory]
            probabilities = reward_impact_ratio / reward_impact_ratio.sum()

            removed_node_index = np.random.choice(trajectory, p=probabilities)
            path[removed_node_index] = -path[removed_node_index]

            trajectory = trajectory[trajectory != removed_node_index]
            if len(positive_indices) == 0:
                break

            total_length = self._calculate_path_length(trajectory, distmx)

        return path

    def get_path_length(self, path: np.ndarray, distmx: np.ndarray) -> float:
        """
        Calculate the length of a specific path.
        
        Args:
            path: Agent's path array
            distmx: Distance matrix between nodes
            
        Returns:
            Total path length
        """
        positive_indices = np.where(path > 0)[0]
        trajectory = positive_indices[np.argsort(path[positive_indices])]
        return self._calculate_path_length(trajectory, distmx)

    def _calculate_removal_impacts(self, trajectory: np.ndarray, distmx: np.ndarray) -> np.ndarray:
        """
        Calculate the impact on path length of removing each node in the trajectory.
        
        Args:
            trajectory: Current trajectory
            distmx: Distance matrix between nodes
            
        Returns:
            Array of removal impacts for each node
            
        Raises:
            ValueError: If any impact is negative
        """
        impacts = np.zeros(len(trajectory))
        
        for i, node_index in enumerate(trajectory):
            impacts[i] = self._calculate_single_node_impact(trajectory, i, node_index, distmx)
            
            if impacts[i] < 0:
                raise ValueError("Impacts must be greater than or equal to zero.")

        return impacts
    
    def _calculate_single_node_impact(
        self, trajectory: np.ndarray, position: int, node_index: int, distmx: np.ndarray
    ) -> float:
        """
        Calculate the impact of removing a single node from the trajectory.
        
        Args:
            trajectory: Current trajectory
            position: Position of node in trajectory
            node_index: Index of the node
            distmx: Distance matrix
            
        Returns:
            Impact value (reduction in path length)
        """
        if position == 0:  # First node
            if len(trajectory) == 1:
                # Only node in path
                original_length = distmx[Solution.begin, node_index] + distmx[node_index, Solution.end]
                new_length = distmx[Solution.begin, Solution.end]
                return original_length - new_length
            else:
                # First of multiple nodes
                next_node = trajectory[1]
                original_length = distmx[Solution.begin, node_index] + distmx[node_index, next_node]
                new_length = distmx[Solution.begin, next_node]
                return original_length - new_length
                
        elif position == len(trajectory) - 1:  # Last node
            prev_node = trajectory[position - 1]
            original_length = distmx[prev_node, node_index] + distmx[node_index, Solution.end]
            new_length = distmx[prev_node, Solution.end]
            return original_length - new_length
            
        else:  # Middle node
            prev_node = trajectory[position - 1]
            next_node = trajectory[position + 1]
            original_length = distmx[prev_node, node_index] + distmx[node_index, next_node]
            new_length = distmx[prev_node, next_node]
            return original_length - new_length + 1e-6

    def _calculate_path_length(self, trajectory: np.ndarray, distmx: np.ndarray) -> float:
        """
        Calculate the total length of a trajectory including start and end nodes.
        
        Args:
            trajectory: Array of node indices in order
            distmx: Distance matrix between nodes
            
        Returns:
            Total trajectory length
        """
        if len(trajectory) == 0:
            return distmx[Solution.begin, Solution.end]

        internal_length = 0.0
        if len(trajectory) > 1:
            internal_length = np.sum(distmx[trajectory[:-1], trajectory[1:]])

        start_to_first = distmx[Solution.begin, trajectory[0]]
        last_to_end = distmx[trajectory[-1], Solution.end]

        return internal_length + start_to_first + last_to_end
    
    def _get_sorted_indices(self, path: np.ndarray) -> np.ndarray:
        """
        Extract and sort the visited nodes from an agent's path.
        
        Args:
            path: Agent's path array
            
        Returns:
            Sorted indices of visited nodes
        """
        positive_indices = np.where(path > 0)[0]
        return positive_indices[np.argsort(path[positive_indices])]

    def copy(self) -> "Solution":
        """
        Create a deep copy of this solution.
        
        Returns:
            A new Solution instance with copied data
        """
        return Solution(
            distmx=None,
            rvalues=None,
            paths=np.copy(self.paths),
            score=copy.deepcopy(self.score),
        )
    
    def __str__(self) -> str:
        """
        String representation of the solution.
        
        Returns:
            Formatted string showing score and paths
        """
        return f"Solution: {self.score}\n {self.paths}"
