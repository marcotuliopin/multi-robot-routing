import copy
import numpy as np


class Solution:
    _BEGIN: int = 0
    _END: int = 0
    _BUDGET: float = 0
    _NUM_AGENTS: int = 0

    def __init__(
        self,
        distmx: np.ndarray,
        rvalues: np.ndarray,
        val: np.ndarray = None,
        score: tuple = (-1, -1),
    ) -> None:
        self.paths = val if val else self.init_paths(distmx.shape[0])
        self.paths = self.bound_solution(self.paths, distmx, rvalues)
        self.score = score
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

    def get_solution_paths(self) -> np.ndarray:
        positive_paths = np.where(self.paths > 0, self.paths, 0)  # Zera valores negativos
        trajectories = positive_paths[positive_paths > 0].reshape(-1, self.paths.shape[1])
        solution = np.hstack([
            np.full((trajectories.shape[0], 1), Solution._BEGIN), 
            trajectories, 
            np.full((trajectories.shape[0], 1), Solution._END)
        ])
        return solution

    def get_solution_length(self, distmx: np.ndarray) -> float:
        paths = self.get_solution_paths()
        lengths = np.sum(distmx[paths[:, :-1], paths[:, 1:]], axis=1)
        return lengths

    def bound_solution(
        self, unbounded_paths: np.ndarray, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        return [self.bound_path(path, distmx, rvalues) for path in unbounded_paths]

    def bound_path(
        self, path: np.ndarray, distmx: np.ndarray, rvalues: np.ndarray
    ) -> np.ndarray:
        positive_indices = np.where(path > 0)[0]
        if len(positive_indices) == 0:
            return path

        total_length = self.__update_path_length(path, positive_indices, distmx)

        probabilities = np.zeros_like(path, dtype=float)
        probabilities[positive_indices] = 1 / rvalues[positive_indices]
        probabilities /= probabilities.sum()

        while total_length > Solution._BUDGET:
            removed_node_index = np.random.choice(
                positive_indices, p=probabilities[positive_indices]
            )
            path[removed_node_index] = -path[removed_node_index]

            positive_indices = positive_indices[positive_indices != removed_node_index]
            if len(positive_indices) == 0:
                break

            probabilities[removed_node_index] = 0
            probabilities /= probabilities.sum()

            total_length = self.__update_path_length(path, positive_indices, distmx)

        return path

    def __update_path_length(
        self, path: np.ndarray, positive_indices: list, distmx: np.ndarray
    ) -> float:
        trajectory = path[positive_indices]
        trajectory.sort()
        total_length = (
            np.sum(distmx[trajectory[:-1], trajectory[1:]])
            + distmx[Solution._BEGIN, trajectory[0]]
            + distmx[trajectory[-1], Solution._END]
        )
        return total_length

    def copy(self) -> "Solution":
        return Solution([val.copy() for val in self.paths], copy.deepcopy(self.score))
