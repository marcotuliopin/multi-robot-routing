import copy
import numpy as np


class Solution:
    _BEGIN: int = 0
    _END: int = 0
    _BUDGET: float = 0

    def __init__(self, val: list[np.ndarray], score: tuple = (-1, -1)) -> None:
        self.unbounded_paths = val  # It is called unbounded_paths because it is not guaranteed to obey the budget constraint
        self.score = score
        self.crowding_distance = -1
        self.visited = False

    # Time complexity: O(1), as it sets class-level parameters.
    # Space complexity: O(1), as it uses a constant amount of additional space.
    @classmethod
    def set_parameters(cls, begin: int, end: int, budget: float) -> None:
        cls._BEGIN = begin
        cls._END = end
        cls._BUDGET = budget

    # Time complexity: O(n), where n is the number of score elements.
    # Space complexity: O(1), as it uses a constant amount of additional space.
    def dominates(self, other: "Solution") -> bool:
        """Check if the current solution dominates the other solution."""
        is_better = False
        for i in range(len(self.score)):
            if self.score[i] > other.score[i]:
                is_better = True
            elif self.score[i] < other.score[i]:
                return False
        return is_better

    # Time complexity: O(n * m), where n is the number of paths and m is the number of points in each path.
    # Space complexity: O(n * m), as it stores the bounded paths.
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

    # Time complexity: O(n * m), where n is the number of paths and m is the number of points in each path.
    # Space complexity: O(n), as it stores the lengths of the paths.
    def get_solution_length(self, distmx: np.ndarray) -> float:
        """Get the length of the solution that obeys the budget constraint."""
        paths = self.get_solution_paths(distmx)
        lengths = []
        for path in paths:
            lengths.append(np.sum(distmx[path[:-1], path[1:]]))
        return lengths

    # Time complexity: O(n), where n is the number of points in the path.
    # Space complexity: O(1), as it uses a constant amount of additional space.
    def bound_solution(self, val: np.ndarray, distmx: np.ndarray) -> np.ndarray:
        """Bound one element of the solution to obey the budget constraint."""
        total = 0
        for idx, (previous, current) in enumerate(zip(val[:-1], val[1:])):
            total += distmx[previous, current]
            if total + distmx[current, Solution._END] > Solution._BUDGET:
                return val[: idx + 1]

        return val

    # Time complexity: O(n * m), where n is the number of paths and m is the number of points in each path.
    # Space complexity: O(n * m), as it stores the copied paths.
    def copy(self) -> "Solution":
        """Create a copy of the current solution."""
        return Solution(
            [val.copy() for val in self.unbounded_paths], copy.deepcopy(self.score)
        )

    # Time complexity: O(1), as it calculates the hash of the solution.
    # Space complexity: O(1), as it uses a constant amount of additional space.
    def __hash__(self):
        return hash(tuple(map(tuple, self.unbounded_paths)))

    # Time complexity: O(n * m), where n is the number of paths and m is the number of points in each path.
    # Space complexity: O(1), as it uses a constant amount of additional space.
    def __eq__(self, other):
        return (
            all(
                np.array_equal(up1, up2)
                for up1, up2 in zip(self.unbounded_paths, other.unbounded_paths)
            )
            and self.score == other.score
        )
