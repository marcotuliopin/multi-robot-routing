import numpy as np


class Solution:
    _BEGIN: int = 0
    _END: int = 0
    _BUDGET: float = 0

    def __init__(self, val: list[np.ndarray], score: tuple = (-1, -1)) -> None:
        self.unbounded_paths = val  # It is called unbounded_paths because it is not guaranteed to obey the budget constraint
        self.score = score
        self.crowding_distance = -1

    @classmethod
    def set_parameters(cls, begin: int, end: int, budget: float) -> None:
        cls._BEGIN = begin
        cls._END = end
        cls._BUDGET = budget

    def dominates(self, other: "Solution") -> bool:
        """Check if the current solution dominates the other solution."""
        is_better = False
        for i in range(len(self.score)):
            if self.score[i] > other.score[i]:
                is_better = True
            elif self.score[i] < other.score[i]:
                return False
        return is_better

    def get_solution_paths(self, distmx: np.ndarray) -> list[np.ndarray]:
        """Get the solution paths that obeys the budget constraint."""
        solution = []
        for path in self.unbounded_paths:
            new_path = path.copy()
            unbounded_val = np.insert(new_path, 0, Solution._BEGIN)
            bounded_val = self.bound_solution(unbounded_val, distmx)
            bounded_val = np.append(bounded_val, Solution._END)
            solution.append(bounded_val)
        return solution

    def get_solution_length(self, distmx: np.ndarray) -> float:
        """Get the length of the solution that obeys the budget constraint."""
        paths = self.get_solution_paths(distmx)
        lengths = []
        for path in paths:
            lengths.append(np.sum(distmx[path[:-1], path[1:]]))
        return lengths

    def bound_solution(self, val: np.ndarray, distmx: np.ndarray) -> np.ndarray:
        """Bound one element of the solution to obey the budget constraint."""
        total = 0
        for idx, (previous, current) in enumerate(zip(val[:-1], val[1:])):
            total += distmx[previous, current]
            if total + distmx[current, Solution._END] > Solution._BUDGET:
                return val[:idx + 1]

        return val

    def copy(self) -> "Solution":
        """Create a copy of the current solution."""
        return Solution([val.copy() for val in self.unbounded_paths], self.score)
