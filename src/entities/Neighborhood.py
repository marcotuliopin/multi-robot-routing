from itertools import combinations
import numpy as np
from . import Solution


class Neighborhood:
    def __init__(self):
        self.perturbation_operators = [
            self.two_opt_all_paths,
            self.invert_points_all_agents,
            self.swap_subpaths_all_agents
        ]
        self.local_search_operators = [
            self.move_point,
            self.invert_single_point,
            self.swap_points,
            self.invert_multiple_points,
            self.two_opt,
            self.path_relinking
        ]

        self.num_neighborhoods = len(self.local_search_operators) * len(self.perturbation_operators)
    
    def get_perturbation_operator(self):
        def wrapper(solution: Solution, rpositions: np.ndarray, max_distance: np.ndarray) -> Solution:
            operator = np.random.randint(0, len(self.perturbation_operators))
            if operator == len(self.perturbation_operators) - 1:
                return self.insert_nearby_rewards(solution, rpositions, max_distance)
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

        return new_solution
    
    def untangle_path(self, solution: Solution) -> Solution:
        def intersect(a, b, c, d):
            def ccw(p1, p2, p3):
                return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        new_solution = solution.copy()
        new_paths = new_solution.paths
        
        for path in new_paths:
            positive_indices = np.where(path > 0)[0]
            sorted_indices = positive_indices[np.argsort(path[positive_indices])]
            positions = solution.rpositions[sorted_indices]

            for i, j in combinations(range(len(sorted_indices) - 1), 2):
                a, b = positions[i], positions[i + 1]
                c, d = positions[j], positions[j + 1]
                
                if intersect(a, b, c, d):
                    new_path_values = path[sorted_indices[i+1:j+1]][::-1]
                    path[sorted_indices[i+1:j+1]] = new_path_values
                
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

        return new_solution

    def invert_points_all_agents(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        for i in range(len(new_paths)):
            for j in range(len(new_paths[i])):
                if np.random.rand() < 0.5:
                    new_paths[i][j] = -new_paths[i][j]

        return new_solution

    def move_towards_order_centroid(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        num_agents = len(new_solution.paths)
        
        path_matrix = np.array([path for path in new_solution.paths])  
        visited_mask = path_matrix > 0  
        visitation_count = np.sum(visited_mask, axis=0)
        centroid_order = np.where(visitation_count > 0, np.sum(path_matrix * visited_mask, axis=0) / visitation_count, -1)

        for agent in range(num_agents):
            path = new_solution.paths[agent]
            for reward in np.where(path > 0)[0]:
                path[reward] = (path[reward] + centroid_order[reward]) / 2
        
        return new_solution
    
    def insert_nearby_rewards(self, solution: Solution, rpositions: np.ndarray, max_distance: float = 5.0) -> Solution:
        new_solution = solution.copy()
        
        for agent in range(len(new_solution.paths)):
            path = new_solution.paths[agent]
            visited_rewards = np.where(path > 0)[0]
            
            for reward in visited_rewards:
                for candidate in range(len(path)):
                    if path[candidate] < 0:  # Not visited
                        distance = np.linalg.norm(rpositions[reward] - rpositions[candidate])
                        if distance < max_distance:
                            insertion_point = path[reward] + 0.5
                            path[candidate] = insertion_point  # Add nearby reward
        
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
                neighbors.append(new_solution)

        return neighbors
    
    def swap_points_all_paths(self, solution: Solution, agent: int) -> list[Solution]:
        neighbors = []

        positive_indices = [np.where(path > 0)[0] for path in solution.paths]
        for _ in range(len(solution.paths[agent])):
            new_solution = solution.copy()
            new_paths = new_solution.paths

            for a in range(len(solution.paths)):
                path = new_paths[a]

                if len(positive_indices[a]) < 2:
                    continue

                i, j = np.random.choice(positive_indices[a], 2, replace=False)
                path[i], path[j] = path[j], path[i]

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

                neighbors.append(new_solution)

        return neighbors

    def invert_single_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(path)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            new_path[i] = -new_path[i]

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

            neighbors.append(new_solution)

        return neighbors
    
    def path_relinking(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        new_solution = solution.copy()

        for i in range(len(solution.paths)):
            if i == agent:
                continue

            indices_to_change = np.random.choice(
                range(len(path)), 
                size=np.random.randint(3, len(path)),
                replace=False
            )

            new_path = new_solution.paths[i]

            for idx in indices_to_change:
                if new_path[idx] == path[idx]:
                    continue
                new_path[idx] = path[idx]
                
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

            neighbors.append(new_solution)

        return neighbors
