from numba import njit
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
        operator = np.random.randint(0, len(self.perturbation_operators))
        return self.perturbation_operators[operator]

    def get_local_search_operator(self, neighborhood: int):
        return self.local_search_operators[neighborhood % len(self.local_search_operators)]

    def two_opt_all_paths(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_solution.paths = two_opt_all_paths_core(solution.paths)
        return new_solution

    def invert_points_all_agents(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_solution.paths = invert_points_all_agents_core(solution.paths)
        return new_solution
    
    def invert_points_all_agents_unique(self, solution: Solution) -> Solution:
        new_solution = solution.copy()
        new_solution.paths = invert_points_all_agents_unique_core(solution.paths)
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


@njit
def copy_paths(paths):
    return paths.copy()


@njit
def get_positive_indices(path):
    return np.where(path > 0)[0]


@njit
def get_negative_indices(path):
    return np.where(path < 0)[0]


@njit
def two_opt_all_paths_core(paths):
    new_paths = copy_paths(paths)
    
    for i in range(len(new_paths)):
        path_len = len(new_paths[i])
        if path_len <= 1:
            continue
            
        idx1 = np.random.randint(0, path_len - 1)
        idx2 = np.random.randint(0, path_len - 1)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        new_paths[i][idx1:idx2 + 1] = new_paths[i][idx1:idx2 + 1][::-1]
    
    return new_paths


@njit
def invert_points_all_agents_core(paths):
    new_paths = copy_paths(paths)
    path_length = new_paths.shape[1]
    
    for i in range(len(new_paths)):
        cols_to_invert = np.random.random(path_length) < 0.75
        new_paths[i][cols_to_invert] *= -1
    
    return new_paths


@njit
def invert_points_all_agents_unique_core(paths):
    new_paths = copy_paths(paths)
    path_length = new_paths.shape[1]
    
    cols_to_invert = np.random.random(path_length) < 0.75
    
    for col in np.where(cols_to_invert)[0]:
        positive_agents = np.where(new_paths[:, col] > 0)[0]
        if len(positive_agents) > 0:
            new_paths[positive_agents, col] *= -1
        
        new_positive_agent = np.random.randint(0, len(new_paths))
        new_paths[new_positive_agent, col] = -new_paths[new_positive_agent, col]
    
    return new_paths
