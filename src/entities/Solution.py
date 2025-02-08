import copy
import numpy as np


class Solution:
    begin: int = -1
    end: int = -1
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
        self.score = score
        self.crowding_distance = -1
        self.visited = False

        if paths is None:
            self.paths = self.init_paths(len(rvalues) - 1)
            self.paths = self.bound_all_paths(self.paths, distmx, rvalues)
        else:
            self.paths = paths
        
    @classmethod
    def set_parameters(
        cls, begin: int, end: int, num_agents: int, budget: list[int], speeds: list[float]
    ) -> None:
        cls.begin = begin
        cls.end = end
        cls.num_agents = num_agents
        cls.budget = budget
        cls.speeds = speeds

    def init_paths(self, num_rewards: int) -> list[np.ndarray]:
        paths = np.random.uniform(
            low=0, high=100, size=(Solution.num_agents, num_rewards)
        )
        
        if any(len(np.unique(path)) != len(path) for path in paths):
            raise ValueError("Paths must have unique values.")

        return paths

    def dominates(self, other: "Solution") -> bool:
        is_better = True
        for i in range(len(self.score)):
            if self.score[i] < other.score[i]:
                return False
        return is_better

    def get_solution_paths(self) -> list[np.ndarray]:
        trajectories = [self.__get_sorted_indices(path) for path in self.paths]
        solution = [
            np.concatenate(([Solution.begin], trajectory, [Solution.end]))
            for trajectory in trajectories
        ]
        return solution

    def get_solution_length(self, distmx: np.ndarray) -> float:
        paths = self.get_solution_paths()
        lengths = np.sum(distmx[paths[:, :-1], paths[:, 1:]], axis=1)
        return lengths

    def bound_all_paths(
        self, paths: np.ndarray, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        for i in range(len(paths)):
            paths[i] = self.bound_path(paths[i], Solution.budget[i], distmx, rvalues)
        return paths

    def bound_path(
    self, path: np.ndarray, budget: int, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        positive_indices = np.where(path > 0)[0]
        trajectory = positive_indices[np.argsort(path[positive_indices])]

        if len(positive_indices) == 0:
            return path

        total_length = self.__update_path_length(path, trajectory, distmx)

        while total_length > budget:
            impacts = self.__get_impact_in_length(trajectory, distmx)

            # The chance of removing a node is proportional to the impact of removing it, but inversely proportional to the reward of the node.
            # Since the reward is always positive, the ratio is always a real number greater than zero.
            reward_impact_ratio = impacts / rvalues[trajectory]
            probabilities = reward_impact_ratio / reward_impact_ratio.sum()

            removed_node_index = np.random.choice(trajectory, p=probabilities)
            path[removed_node_index] = -path[removed_node_index]

            trajectory = trajectory[trajectory != removed_node_index]
            if len(positive_indices) == 0:
                break

            total_length = self.__update_path_length(path, trajectory, distmx)

        return path

    def get_path_length(self, path: np.ndarray, distmx: np.ndarray) -> float:
        positive_indices = np.where(path > 0)[0]
        return self.__update_path_length(path, positive_indices, distmx)
    
    def __get_impact_in_length(self, trajectory: np.ndarray, distmx: np.ndarray) -> np.ndarray:
        """
        Calculate the impact in length of removing each node in the path.
        """
        impacts = np.zeros(len(trajectory))
        for i, index in enumerate(trajectory):
            # If the node is the first node in the path.
            if i == 0:
                # If the path has only one node, the impact is zeroing the length.
                if len(trajectory) == 1:
                    impacts[i] = distmx[Solution.begin, index]
                # If the path has more than one node, the impact is the difference between the length of the path and the length of the path without the first node.
                elif len(trajectory) > 1:
                    original_length = distmx[Solution.begin, index] + distmx[index, trajectory[1]]
                    impacts[i] = original_length - distmx[Solution.begin, trajectory[1]]
            # If the node is the last node and at the same time not the first, this means the path is of at least two nodes.
            elif i == len(trajectory) - 1:
                # The impact is the difference between the length of the path and the length of the path without the last node.
                original_length = distmx[trajectory[i - 1], index] + distmx[index, Solution.end]
                impacts[i] = original_length - distmx[trajectory[i - 1], Solution.end]
            # If the node is in the middle of the path and is neither the first nor the last node, then the path is of at least three nodes.
            else:
                # The impact is the difference between the length of the path and the length of the path without the node.
                original_length = distmx[trajectory[i - 1], index] + distmx[index, trajectory[i + 1]]
                impacts[i] = original_length - distmx[trajectory[i - 1], trajectory[i + 1]]

        return impacts

    def __update_path_length(self, path: np.ndarray, trajectory: np.ndarray, distmx: np.ndarray) -> float:
        total_length = (
            np.sum(distmx[trajectory[:-1], trajectory[1:]])
            + distmx[Solution.begin, trajectory[0]]
            + distmx[trajectory[-1], Solution.end]
        )
        return total_length
    
    def __get_sorted_indices(self, path: np.ndarray) -> np.ndarray:
        positive_indices = np.where(path > 0)[0]
        return positive_indices[np.argsort(path[positive_indices])]

    def copy(self) -> "Solution":
        return Solution(
            distmx=None,
            rvalues=None,
            paths=np.copy(self.paths),
            score=copy.deepcopy(self.score),
        )
    
    def __str__(self) -> str:
        return f"Solution: {self.score}\n {self.paths}"
