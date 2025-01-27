import numpy as np
from . import Solution


class Neighborhood:
    def __init__(self):
        self.perturbation_operators = {
            0: self.two_opt,
            1: self.swap_all_subpaths,
        }
        self.local_search_operators = {
            0: self.move_point,
            1: self.invert_single_point,
            2: self.swap_points,
            3: self.invert_multiple_points,
            4: self.swap_local_subpaths,
            5: self.move_point,
            6: self.invert_single_point,
            7: self.swap_points,
            8: self.invert_multiple_points,
            9: self.swap_local_subpaths,
        }
        self.num_neighborhoods = len(self.local_search_operators)
    
    def get_perturbation_operator(self, neighborhood: int):
        return self.perturbation_operators[neighborhood % len(self.perturbation_operators)]

    def get_local_search_operator(self, neighborhood: int):
        return self.local_search_operators[neighborhood % len(self.local_search_operators)]

    def two_opt(solution: Solution) -> list:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        i, j = np.random.choice(len(new_paths[0]) - 1, 2, replace=False)
        if i > j:
            i, j = j, i

        for new_path in new_paths:
            new_path = np.concatenate(
                [new_path[:i], new_path[i : j + 1][::-1], new_path[j + 1 :]]
            )

        return new_solution

    def swap_all_subpaths(solution: Solution) -> np.ndarray:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        l = np.random.randint(1, len(new_paths[0]) // 2)
        i = np.random.randint(0, len(new_paths[0]) - 2 * l)
        j = np.random.randint(i + l, len(new_paths[0]) - l)

        for new_path in new_paths:
            new_path[i : i + l], new_path[j : j + l] = (
                new_path[j : j + l].copy(),
                new_path[i : i + l].copy(),
            )

        return new_solution

    def move_point(solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        positive_indices = np.where(path > 0)[0]

        for i in range(len(positive_indices)):
            for j in range(i + 1, len(positive_indices)):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]

                idx1 = positive_indices[i]
                idx2 = positive_indices[j]

                if j == len(positive_indices) - 1:
                    # The new position is the last point
                    new_path[idx1] = new_path[idx2]
                    # The last point is the middle point between the two points
                    new_path[idx2] = (
                        new_path[idx1]
                        + (new_path[positive_indices[j - 1]] - new_path[idx1]) / 2
                    )
                else:
                    # The new position is the middle point between the two points
                    new_path[idx1] = (
                        new_path[idx2]
                        + (new_path[positive_indices[j + 1]] - new_path[idx2]) / 2
                    )

                neighbors.append(new_solution)

        return neighbors

    def swap_points(solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        positive_indices = np.where(path > 0)[0]

        for i in range(len(positive_indices)):
            for j in range(i + 1, len(positive_indices)):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]

                idx1 = positive_indices[i]
                idx2 = positive_indices[j]
                new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]

                neighbors.append(new_solution)

        return neighbors

    def swap_local_subpaths(solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        l = np.random.randint(1, len(path) // 2)

        for i in range(len(path) - 2 * l):
            for j in range(i + l, len(path) - l):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]

                new_path[i : i + l], new_path[j : j + l] = (
                    new_path[j : j + l].copy(),
                    new_path[i : i + l].copy(),
                )

                neighbors.append(new_solution)

        return neighbors

    def invert_single_point(solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(path)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            new_path[i] = -new_path[i]

            neighbors.append(new_solution)

        return neighbors

    def invert_multiple_points(solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(1, len(path) + 1):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            idxs = np.random.choice(len(new_path), i, replace=False)
            new_path[idxs] = -new_path[idxs]

            neighbors.append(new_solution)

        return neighbors
