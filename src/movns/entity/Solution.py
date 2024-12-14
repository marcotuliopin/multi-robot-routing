import numpy as np


class Solution:
    _BEGIN: int = 0
    _END: int = 0
    _BUDGET: float = 0

    def __init__(self, val: list[np.ndarray], score: float = None) -> None:
        self.unbound_solution = val  # It is called _unbound_solution because it is not guaranteed to obey the budget constraint
        self.score = score
        self.crowding_distance = -1

    @classmethod
    def set_parameters(cls, begin: int, end: int, budget: float) -> None:
        cls._BEGIN = begin
        cls._END = end
        cls._BUDGET = budget

    def dominates(self, other: "Solution") -> bool:
        """Check if the current solution dominates the other solution."""
        return all(self.score[i] >= other.score[i] for i in range(len(self.score)))

    def get_solution(self, distmx: np.ndarray) -> list[np.ndarray]:
        """Get the solution that obeys the budget constraint."""
        solution = []
        for path in self.unbound_solution:
            untrimmed_val = np.insert(path, 0, Solution._BEGIN)
            trimmed_val = self.trim_solution(untrimmed_val, distmx)
            trimmed_val = np.append(trimmed_val, Solution._END)
            solution.append(trimmed_val)
        return solution

    def get_solution_length(self, distmx: np.ndarray) -> float:
        """Get the length of the solution that obeys the budget constraint."""
        solution = self.get_solution(distmx)
        total_lengths = 0
        for path in solution:
            total_lengths += np.sum(distmx[path[:-1], path[1:]])
        return total_lengths

    def trim_solution(self, val: np.ndarray, distmx: np.ndarray) -> np.ndarray:
        """Trim one element of the solution to obey the budget constraint."""
        total = 0
        previous = Solution._BEGIN

        for i in range(len(val)):
            current = val[i]
            total += distmx[previous, current]
            if total + distmx[current, Solution._END] > Solution._BUDGET:
                return val[:i]
            previous = current

        return val

    def copy(self) -> "Solution":
        """Create a copy of the current solution."""
        return Solution([val.copy() for val in self.unbound_solution], self.score)
