import copy
import numpy as np


class Solution:
    _BEGIN: int = 0
    _END: int = 0
    _BUDGET: float = 0
    _NUM_AGENTS: int = 0

    def __init__(
        self,
        distmx: np.ndarray = None,
        rvalues: np.ndarray = None,
        val: np.ndarray = None,
        score: tuple = None,
    ) -> None:
        if val is None:
            self.paths = self.init_paths(distmx.shape[0])
            self.paths = self.bound_all_paths(self.paths, distmx, rvalues)
        else:
            self.paths = val

        self.score = score if not score is None else (-1, -1)
        self.crowding_distance = -1
        self.visited = False

    @classmethod
    def set_parameters(
        cls, begin: int, end: int, budget: float, num_agents: int
    ) -> None:
        cls._BEGIN = begin
        cls._END = end
        cls._BUDGET = budget
        cls._NUM_AGENTS = num_agents

    def init_paths(self, num_rewards: int) -> list[np.ndarray]:
        return np.random.uniform(
            low=-1, high=1, size=(Solution._NUM_AGENTS, num_rewards)
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
            np.concatenate(([Solution._BEGIN], trajectory, [Solution._END]))
            for trajectory in trajectories
        ]
        return solution

    def get_solution_length(self, distmx: np.ndarray) -> float:
        paths = self.get_solution_paths()
        lengths = np.sum(distmx[paths[:, :-1], paths[:, 1:]], axis=1)
        return lengths

    def bound_all_paths(
        self, unbounded_paths: np.ndarray, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        return np.apply_along_axis(self.bound_path, 1, unbounded_paths, distmx, rvalues)

    def bound_path(
        self, path: np.ndarray, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        positive_indices = np.where(path > 0)[0]

        if len(positive_indices) == 0:
            return path

        total_length = self.__update_path_length(path, positive_indices, distmx)

        if total_length <= Solution._BUDGET:
            return path

        probabilities = np.zeros_like(path, dtype=float)
        probabilities[positive_indices] = 1 / rvalues[positive_indices]

        probabilities = probabilities / probabilities.sum()

        while total_length > Solution._BUDGET:
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
            + distmx[Solution._BEGIN, trajectory[0]]
            + distmx[trajectory[-1], Solution._END]
        )
        return total_length
    
    def __get_sorted_indices(self, path: np.ndarray) -> np.ndarray:
        positive_indices = np.where(path > 0)[0]
        return positive_indices[np.argsort(path[positive_indices])]

    def copy(self) -> "Solution":
        return Solution(
            distmx=None,
            rvalues=None,
            val=np.copy(self.paths),
            score=copy.deepcopy(self.score),
        )
    
    def __str__(self) -> str:
        return f"Solution: {self.score}\n {self.paths}"
