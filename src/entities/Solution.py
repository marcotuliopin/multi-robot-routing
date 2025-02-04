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
        score: tuple = (-1, -1),
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
        return np.random.uniform(
            low=0, high=1, size=(Solution.num_agents, num_rewards)
        )

    def dominates(self, other: "Solution") -> bool:
        is_better = False
        for i in range(len(self.score)):
            if self.score[i] > other.score[i]:
                is_better = True
            elif self.score[i] < other.score[i]:
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

        if len(positive_indices) == 0:
            return path

        total_length = self.__update_path_length(path, positive_indices, distmx)

        if total_length <= budget:
            return path

        probabilities = np.zeros_like(path, dtype=float)
        probabilities[positive_indices] = 1 / rvalues[positive_indices]

        probabilities = probabilities / probabilities.sum()

        while total_length > budget:
            removed_node_index = np.random.choice(
                positive_indices, p=probabilities[positive_indices]
            )
            path[removed_node_index] = -path[removed_node_index]

            positive_indices = positive_indices[positive_indices != removed_node_index]
            if len(positive_indices) == 0:
                break

            probabilities[removed_node_index] = 0
            probabilities = probabilities / probabilities.sum()

            total_length = self.__update_path_length(path, positive_indices, distmx)

        return path

    def get_path_length(self, path: np.ndarray, distmx: np.ndarray) -> float:
        positive_indices = np.where(path > 0)[0]
        return self.__update_path_length(path, positive_indices, distmx)

    def __update_path_length(
        self, path: np.ndarray, positive_indices: list, distmx: np.ndarray
    ) -> float:
        trajectory = positive_indices[np.argsort(path[positive_indices])]

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
