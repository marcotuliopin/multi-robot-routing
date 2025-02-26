from itertools import combinations
from . import Solution
from .SanityAssertions import SanityAssertions
import numpy as np


class Neighborhood:
    def __init__(self):
        self.perturbation_operators = [
            self.invert_points_all_agents_unique,
            # self.invert_points_all_agents,
            self.two_opt_all_paths,
            # self.untangle_path,
            self.swap_subpaths_all_agents,
        ]
        self.local_search_operators = [
            self.move_point,
            self.swap_points,
            self.invert_single_point_unique,
            # self.invert_single_point,
            self.add_and_move,
            self.swap_local_subpaths,
            self.remove_point,
            self.two_opt,
            # self.path_relinking,
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

        for new_path in new_paths:
            i, j = np.random.choice(len(new_paths[0]) - 1, 2, replace=False)
            if i > j:
                i, j = j, i
            new_path = np.concatenate(
                [new_path[:i], new_path[i : j + 1][::-1], new_path[j + 1 :]]
            )

        # for new_path in new_paths:
        #     SanityAssertions.assert_no_repeated_values(new_path)

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

        # SanityAssertions.assert_one_agent_per_reward(new_paths)
        # for new_path in new_paths:
        #     SanityAssertions.assert_no_repeated_values(new_path)
                
        return new_solution

    def swap_subpaths_all_agents(self, solution: Solution) -> np.ndarray:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        for new_path in new_paths:
            positive_idxs = np.where(new_path > 0)[0]

            if len(positive_idxs) // 2 < 2:
                continue

            l = np.random.randint(1, len(positive_idxs) // 2)
            i = np.random.randint(0, len(positive_idxs) - 2 * l)
            j = np.random.randint(i + l, len(positive_idxs) - l)

            new_path[positive_idxs[i : i + l]], new_path[positive_idxs[j : j + l]] = (
                new_path[positive_idxs[j : j + l]].copy(),
                new_path[positive_idxs[i : i + l]].copy(),
            )
        
        # SanityAssertions.assert_one_agent_per_reward(new_paths)
        # for new_path in new_paths:
        #     SanityAssertions.assert_no_repeated_values(new_path)

        return new_solution

    def invert_points_all_agents(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        for i in range(len(new_paths)):
            for j in range(len(new_paths[i])):
                if np.random.rand() < 0.75:
                    new_paths[i][j] = -new_paths[i][j]

        # for new_path in new_paths:
        #     SanityAssertions.assert_no_repeated_values(new_path)

        return new_solution
    
    def invert_points_all_agents_unique(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        path_length = len(new_solution.paths[0])

        for col in range(path_length):
            if np.random.random() < 0.75:
                positive_idx = np.where(new_solution.paths[:, col] > 0)[0]
                new_solution.paths[positive_idx, col] = -new_solution.paths[positive_idx, col]

                new_positive_idx = np.random.randint(0, len(new_solution.paths))
                new_solution.paths[new_positive_idx, col] = -new_solution.paths[new_positive_idx, col]

        # SanityAssertions.assert_one_agent_per_reward(new_solution.paths)
        # for new_path in new_paths:
        #     SanityAssertions.assert_no_repeated_values(new_path)

        return new_solution

    def move_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        positive_indices = np.where(path > 0)[0]

        for i in range(len(positive_indices)):
            for j in range(i + 1, len(positive_indices)):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]
                new_path[positive_indices[i]] = new_path[positive_indices[i]] + np.random.random() * self.epsilon

                # SanityAssertions.assert_no_repeated_values(new_path)

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

                # SanityAssertions.assert_one_agent_per_reward(new_solution.paths)
                # SanityAssertions.assert_no_repeated_values(new_path)

                neighbors.append(new_solution)

        return neighbors
    
    def two_opt(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []
        positive_indices = np.where(path > 0)[0]

        for i in range(len(positive_indices) - 1):
            for j in range(i + 1, len(positive_indices)):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]

                new_path[positive_indices[i : j + 1]] = (
                    new_path[positive_indices[i: j + 1]][::-1]
                )

                # SanityAssertions.assert_one_agent_per_reward(new_solution.paths)
                # SanityAssertions.assert_no_repeated_values(new_path)

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

                # SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors

    def swap_local_subpaths(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        positive_idxs = np.where(path > 0)[0]

        if len(positive_idxs) // 2 < 2:
            return neighbors

        l = np.random.randint(1, len(positive_idxs) // 2)

        for i in range(len(positive_idxs) - 2 * l):
            for j in range(i + l, len(positive_idxs) - l):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]

                new_path[positive_idxs[i : i + l]], new_path[positive_idxs[j : j + l]] = (
                    new_path[positive_idxs[j : j + l]].copy(),
                    new_path[positive_idxs[i : i + l]].copy(),
                )

                # SanityAssertions.assert_one_agent_per_reward(new_solution.paths)
                # SanityAssertions.assert_no_repeated_values(new_path)

                neighbors.append(new_solution)

        return neighbors

    def invert_single_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(path)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            new_path[i] = -new_path[i]

            # SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors

    def invert_single_point_unique(self, solution: Solution, agent: int) -> list[Solution]:
        return self. add_point_unique(solution, agent) + self.remove_point(solution, agent)

    def add_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []
        
        negative_indices = np.where(path < 0)[0]

        for i in range(len(negative_indices)):
            idx = negative_indices[i]
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]
            new_path[idx] = -new_path[idx]

            # SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors
    
    def add_point_unique(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        negative_indices = np.where(path < 0)[0]

        for i in range(len(negative_indices)):
            idx = negative_indices[i]
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]
            
            positive_indices = np.where(new_solution.paths[:, idx] > 0)[0]
            new_solution.paths[positive_indices, idx] = -new_solution.paths[positive_indices, idx]
            new_path[idx] = -new_path[idx]

            # SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)
        
        return neighbors

    def add_and_move(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        negative_indices = np.where(path < 0)[0]
        positive_indices = np.where(path > 0)[0]

        for i in range(len(negative_indices)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            positive_idx = np.where(new_solution.paths[:, negative_indices[i]] > 0)[0]
            new_solution.paths[positive_idx, negative_indices[i]] = -new_solution.paths[positive_idx, negative_indices[i]]

            new_path[negative_indices[i]] = -new_path[negative_indices[i]]

            neighbors.append(new_solution)

            for j in range(len(positive_indices)):
                new_solution = new_solution.copy()
                new_path = new_solution.paths[agent]

                new_path[negative_indices[i]] = new_path[positive_indices[j]] + np.random.random() * self.epsilon

                neighbors.append(new_solution)

        return neighbors

    def add_and_swap(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        negative_indices = np.where(path < 0)[0]
        positive_indices = np.where(path > 0)[0]

        for i in range(len(negative_indices)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            positive_idx = np.where(new_solution.paths[:, negative_indices[i]] > 0)[0]
            new_solution.paths[positive_idx, negative_indices[i]] = -new_solution.paths[positive_idx, negative_indices[i]]

            new_path[negative_indices[i]] = -new_path[negative_indices[i]]

            neighbors.append(new_solution)

            for j in range(len(positive_indices)):
                new_solution = new_solution.copy()
                new_path = new_solution.paths[agent]

                new_path[negative_indices[i]], new_path[positive_indices[j]] = new_path[positive_indices[j]], new_path[negative_indices[i]]

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

            # SanityAssertions.assert_no_repeated_values(new_path)

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

                # SanityAssertions.assert_no_repeated_values(new_path)
                
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

            # SanityAssertions.assert_no_repeated_values(new_path)

            neighbors.append(new_solution)

        return neighbors

