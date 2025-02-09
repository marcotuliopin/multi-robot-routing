from itertools import combinations
from . import Solution
from .SanityAssertions import SanityAssertions
import numpy as np


class Neighborhood:
    def __init__(self):
        self.perturbation_operators = [
            self.two_opt_all_paths,
            self.invert_points_all_agents,
            self.swap_subpaths_all_agents,
            self.untangle_path,
        ]
        self.local_search_operators = [
            self.swap_points,
            self.add_point,
            self.remove_point,
            self.two_opt,
            self.invert_multiple_points,
            self.swap_local_subpaths,
            self.path_relinking,
        ]

        self.num_neighborhoods = len(self.local_search_operators) * len(self.perturbation_operators)
        self.epsilon = 1e-4
    
    def get_perturbation_operator(self):
        def wrapper(solution: Solution, rpositions: np.ndarray) -> Solution:
            operator = np.random.randint(0, len(self.perturbation_operators))
            if operator == len(self.perturbation_operators) - 1:
                return self.untangle_path(solution, rpositions)
            return self.perturbation_operators[operator](solution)
            
        return wrapper

    def get_local_search_operator(self, neighborhood: int):
        return self.local_search_operators[neighborhood % len(self.local_search_operators)]

    def two_opt_all_paths(self, solution: Solution) -> list:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        i, j = np.random.choice(len(new_paths[0]) - 1, 2, replace=False)
        if i > j:
            i, j = j, i

        for new_path in new_paths:
            new_path = np.concatenate(
                [new_path[:i], new_path[i : j + 1][::-1], new_path[j + 1 :]]
            )

        for new_path in new_paths:
            SanityAssertions.assert_no_repeated_values(new_path)

        return new_solution
    
    def untangle_path(self, solution: Solution, rpositions: np.ndarray) -> Solution:
        def intersect(a, b, c, d):
            def ccw(p1, p2, p3):
                return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        new_solution = solution.copy()
        new_paths = new_solution.paths
        
        for path in new_paths:
            positive_indices = np.where(path > 0)[0]
            sorted_indices = positive_indices[np.argsort(path[positive_indices])]
            positions = rpositions[sorted_indices]

            for i, j in combinations(range(len(sorted_indices) - 1), 2):
                a, b = positions[i], positions[i + 1]
                c, d = positions[j], positions[j + 1]
                
                if intersect(a, b, c, d):
                    new_path_values = path[sorted_indices[i+1:j+1]][::-1]
                    path[sorted_indices[i+1:j+1]] = new_path_values

        for new_path in new_paths:
            SanityAssertions.assert_no_repeated_values(new_path)
                
        return new_solution

    def swap_subpaths_all_agents(self, solution: Solution) -> np.ndarray:
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

        for new_path in new_paths:
            SanityAssertions.assert_no_repeated_values(new_path)

        return new_solution

    def invert_points_all_agents(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        for i in range(len(new_paths)):
            for j in range(len(new_paths[i])):
                if np.random.rand() < 0.5:
                    new_paths[i][j] = -new_paths[i][j]

        for new_path in new_paths:
            SanityAssertions.assert_no_repeated_values(new_path)

        return new_solution

    def move_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        positive_indices = np.where(path > 0)[0]

        for i in range(len(positive_indices)):
            for j in range(i + 1, len(positive_indices)):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]

                idx1 = positive_indices[i]
                idx2 = positive_indices[j]

                # If the new position is the last point.
                if j == len(positive_indices) - 1:
                    # The new position is the last point.
                    new_path[idx1] = new_path[idx2]
                    # The last point is the middle point between the two points.
                    new_path[idx2] = (
                        new_path[idx1]
                        + (new_path[positive_indices[j - 1]] - new_path[idx1]) / 2
                    )
                else:
                    # The new position is the middle point between the two points.
                    new_path[idx1] = (
                        new_path[idx2]
                        + (new_path[positive_indices[j + 1]] - new_path[idx2]) / 2
                    )

                SanityAssertions.assert_no_repeated_values(new_path)

                neighbors.append(new_solution)

        return neighbors

    def swap_points(self, solution: Solution, agent: int) -> list[Solution]:
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

                SanityAssertions.assert_no_repeated_values(new_path)

                neighbors.append(new_solution)

        return neighbors
    
    def two_opt(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(path) - 1):
            for j in range(i + 1, len(path)):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]
                new_path = np.concatenate(
                    [new_path[:i], new_path[i : j + 1][::-1], new_path[j + 1 :]]
                )

                SanityAssertions.assert_no_repeated_values(new_path)

                neighbors.append(new_solution)

        return neighbors
    
    def swap_points_all_paths(self, solution: Solution, agent: int) -> list[Solution]:
        neighbors = []

        positive_indices = [np.where(path > 0)[0] for path in solution.paths]
        for _ in range(len(solution.paths[agent])):
            new_solution = solution.copy()
            new_paths = new_solution.paths

            for idx in range(len(solution.paths)):
                new_path = new_paths[idx]

                if len(positive_indices[idx]) < 2:
                    continue

                i, j = np.random.choice(positive_indices[idx], 2, replace=False)
                new_path[i], new_path[j] = new_path[j], new_path[i]

                SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors

    def swap_local_subpaths(self, solution: Solution, agent: int) -> list[Solution]:
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

                SanityAssertions.assert_no_repeated_values(new_path)

                neighbors.append(new_solution)

        return neighbors

    def invert_single_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(path)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            new_path[i] = -new_path[i]

            SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors
    
    def add_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []
        
        negative_indices = np.where(path < 0)[0]

        for i in range(len(negative_indices)):
            idx = negative_indices[i]
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]
            new_path[idx] = -new_path[idx]

            SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors
    
    def remove_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        positive_indices = np.where(path > 0)[0]

        for i in range(len(positive_indices)):
            idx = positive_indices[i]
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]
            new_path[idx] = -new_path[idx]

            SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors
    
    def path_relinking(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(solution.paths)):
            if i == agent:
                continue

            indices_to_change = np.random.choice(
                range(len(path)), 
                size=np.random.randint(1, len(path)),
                replace=False
            )

            new_solution = solution.copy()
            new_path = new_solution.paths[i]

            for idx in indices_to_change:
                if new_path[idx] == path[idx]:
                    continue
                new_path[idx] = path[idx] + np.random.random() * self.epsilon

                SanityAssertions.assert_no_repeated_values(new_path)
                
                neighbors.append(new_solution.copy())

        return neighbors

    def invert_multiple_points(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(1, len(path) + 1):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            idxs = np.random.choice(len(new_path), i, replace=False)
            new_path[idxs] = -new_path[idxs]

            SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors