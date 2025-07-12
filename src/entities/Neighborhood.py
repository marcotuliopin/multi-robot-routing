from itertools import combinations
import numpy as np

from . import Solution


class Neighborhood:
    def __init__(self, algorithm: str = "unique_vis"):
        if algorithm == "unique_vis":
            self.perturbation_operators = [
                self.invert_points_all_agents_unique,
                self.two_opt_all_paths,
            ]
            self.local_search_operators = [
                self.move_point,
                self.swap_points,
                self.invert_single_point_unique,
                self.add_and_move_unique,
                self.two_opt,
            ]
        else:
            self.perturbation_operators = [
                self.invert_points_all_agents,
                self.two_opt_all_paths,
            ]
            self.local_search_operators = [
                self.move_point,
                self.swap_points,
                self.invert_single_point,
                self.add_and_move,
                self.two_opt,
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

    def two_opt_all_paths(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_paths = new_solution.paths

        for new_path in new_paths:
            i, j = np.random.choice(len(new_paths[0]) - 1, 2, replace=False)
            if i > j:
                i, j = j, i
            new_path[i:j + 1] = new_path[i:j + 1][::-1]

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
            if len(positive_indices) < 2:
                continue

            sorted_indices = positive_indices[np.argsort(path[positive_indices])]
            positions = rpositions[sorted_indices]

            for i, j in combinations(range(len(sorted_indices) - 1), 2):
                a, b = positions[i], positions[i + 1]
                c, d = positions[j], positions[j + 1]
                
                if intersect(a, b, c, d):
                    indices_to_reverse = sorted_indices[i+1:j+1]
                    path[indices_to_reverse] = path[indices_to_reverse][::-1].copy()
                
        return new_solution

    def invert_points_all_agents(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_paths = new_solution.paths
        path_length = len(new_paths[0])

        for i in range(len(new_paths)):
            cols_to_invert = np.random.random(path_length) < 0.75
            new_paths[i][cols_to_invert] *= -1

        return new_solution
    
    def invert_points_all_agents_unique(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        path_length = len(new_solution.paths[0])

        cols_to_invert = np.random.random(path_length) < 0.75

        for col in np.where(cols_to_invert)[0]:
            positive_idx = np.where(new_solution.paths[:, col] > 0)[0]
            new_solution.paths[positive_idx, col] *= -1

            new_positive_idx = np.random.randint(0, len(new_solution.paths))
            new_solution.paths[new_positive_idx, col] *= -1

        return new_solution

    def move_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        positive_indices = np.where(path > 0)[0]

        if len(positive_indices) < 2:
            return []

        neighbors = []

        for i in range(len(positive_indices)):
            source_idx = positive_indices[i]
            source_value = path[source_idx]
            
            for j in range(len(positive_indices)):
                if i == j:
                    continue
                    
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]
                
                positive_values = new_path[positive_indices].copy()
                
                positive_values = np.delete(positive_values, i)
                insertion_pos = j if j < i else j - 1
                positive_values = np.insert(positive_values, insertion_pos, source_value)
                
                new_path[positive_indices] = positive_values
                
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

                idx1, idx2 = positive_indices[i], positive_indices[j]
                new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]

                neighbors.append(new_solution)

        return neighbors
    
    def two_opt(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        positive_indices = np.where(path > 0)[0]

        if len(positive_indices) < 2:
            return []

        neighbors = []

        for i in range(len(positive_indices) - 1):
            for j in range(i + 1, len(positive_indices)):
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]

                indices_to_reverse = positive_indices[i:j+1]
                new_path[indices_to_reverse] = new_path[indices_to_reverse][::-1].copy()

                neighbors.append(new_solution)

        return neighbors

    def invert_single_point(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        for i in range(len(path)):
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]
            new_path[i] *= -1
            neighbors.append(new_solution)

        return neighbors

    def invert_single_point_unique(self, solution: Solution, agent: int) -> list[Solution]:
        return self. add_point_unique(solution, agent) + self.remove_point(solution, agent)
    
    def add_point_unique(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        negative_indices = np.where(path < 0)[0]

        for i in negative_indices:
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]
            
            positive_indices = np.where(new_solution.paths[:, i] > 0)[0]
            new_solution.paths[positive_indices, i] *= -1
            new_path[i] *= -1

            neighbors.append(new_solution)
        
        return neighbors

    def add_and_move(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        negative_indices = np.where(path < 0)[0]

        for add_idx in negative_indices:
            # Add
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]
            new_path[add_idx] *= -1

            new_positive_indices = np.where(new_path > 0)[0]
            source_value = new_path[add_idx]

            # Move
            for target_idx in new_positive_indices:
                if add_idx == target_idx:
                    continue
                    
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]
                
                positive_values = new_path[new_positive_indices].copy()
                
                source_rel_idx = np.where(new_positive_indices == add_idx)[0][0]
                target_rel_idx = np.where(new_positive_indices == target_idx)[0][0]
                
                positive_values = np.delete(positive_values, source_rel_idx)
                
                insert_pos = target_rel_idx if target_rel_idx < source_rel_idx else target_rel_idx - 1
                positive_values = np.insert(positive_values, insert_pos, source_value)
                
                new_path[new_positive_indices] = positive_values
                
                neighbors.append(new_solution)

        return neighbors

    def add_and_move_unique(self, solution: Solution, agent: int) -> list[Solution]:
        path = solution.paths[agent]
        neighbors = []

        negative_indices = np.where(path < 0)[0]
        positive_indices = np.where(path > 0)[0]

        for add_idx in negative_indices:
            # Add
            new_solution = solution.copy()
            new_path = new_solution.paths[agent]

            positive_indices = np.where(new_solution.paths[:, add_idx] > 0)[0]
            new_solution.paths[positive_indices, add_idx] *= -1
            new_path[add_idx] *= -1

            new_positive_indices = np.where(new_path > 0)[0]
            source_value = new_path[add_idx]

            # Move
            for target_idx in new_positive_indices:
                if add_idx == target_idx:
                    continue
                    
                new_solution = solution.copy()
                new_path = new_solution.paths[agent]
                
                positive_values = new_path[new_positive_indices].copy()
                
                source_rel_idx = np.where(new_positive_indices == add_idx)[0][0]
                target_rel_idx = np.where(new_positive_indices == target_idx)[0][0]
                
                positive_values = np.delete(positive_values, source_rel_idx)
                
                insert_pos = target_rel_idx if target_rel_idx < source_rel_idx else target_rel_idx - 1
                positive_values = np.insert(positive_values, insert_pos, source_value)
                
                new_path[new_positive_indices] = positive_values
                
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

                neighbors.append(new_solution.copy())

        return neighbors
