import time
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist

import plot
from .evaluation import evaluate, update_archive
from .operators import local_search, perturb_solution
from .entities import Solution, Neighborhood

ARCHIVE_MAX_SIZE = 40
FRONT_SELECTION_PROBABILITY = 0.9
DEFAULT_OUTPUT_DIR = "imgs/"
DEFAULT_PATHS_SUBDIR = "paths/"

def save_statistics(log: list, front: list) -> None:
    """
    Save statistics of the current Pareto front.
    
    Args:
        log: List to store statistics
        front: Current Pareto front solutions
    """
    if not front:
        return
        
    max_reward = max(solution.score[0] for solution in front)
    max_rssi = max(solution.score[1] for solution in front)
    log.append([max_reward, max_rssi])


def select_solution_from_front(front: list, dominated: list, iteration: int) -> Solution:
    """
    Select a solution from the front or dominated set based on iteration strategy.
    
    Args:
        front: Pareto front solutions
        dominated: Dominated solutions
        iteration: Current iteration number
        
    Returns:
        Selected solution
    """
    # Decide whether to use front or dominated solutions
    use_front = np.random.random() < FRONT_SELECTION_PROBABILITY or not dominated
    candidates = _get_available_candidates(front if use_front else dominated)
    
    # Select based on iteration parity
    if iteration % 2 == 0:
        solution = _select_by_reward(candidates)
    else:
        solution = _select_by_rssi(candidates)
    
    solution.visited = True
    return solution


def _get_available_candidates(solutions: list) -> list:
    """
    Get unvisited solutions or reset all if none available.
    
    Args:
        solutions: List of solutions to check
        
    Returns:
        List of available candidate solutions
    """
    candidates = [s for s in solutions if not s.visited]
    
    if not candidates:
        # Reset all solutions if no unvisited ones remain
        for solution in solutions:
            solution.visited = False
        candidates = solutions
        
    return candidates


def _select_by_reward(candidates: list) -> Solution:
    """Select solution probabilistically based on reward scores."""
    rewards = np.array([s.score[0] for s in candidates])
    probabilities = rewards / np.sum(rewards)
    return np.random.choice(candidates, p=probabilities)


def _select_by_rssi(candidates: list) -> Solution:
    """Select solution probabilistically based on RSSI scores."""
    rssi_scores = np.array([1 / s.score[1] for s in candidates])
    probabilities = rssi_scores / np.sum(rssi_scores)
    return np.random.choice(candidates, p=probabilities)

# ==================== MAIN ALGORITHM FUNCTIONS ====================


def run_movns(
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray,
    total_time: int,
    seed: int,
    max_iterations: int = 100,
    algorithm: str = "unique_vis",
) -> tuple[list, list, list]:
    """
    Run the Multi-Objective Variable Neighborhood Search algorithm.
    
    Args:
        rvalues: Reward values for each node
        rpositions: Positions of reward nodes
        distmx: Distance matrix between nodes
        total_time: Maximum execution time in seconds
        seed: Random seed for reproducibility
        max_iterations: Maximum number of iterations
        algorithm: Algorithm variant to use
        
    Returns:
        Tuple of (archive, front, log)
    """
    np.random.seed(seed)

    neighborhood = Neighborhood(algorithm)
    
    # Initialize with a random solution
    initial_solution = Solution(distmx=distmx, rvalues=rvalues)
    initial_solution.score = evaluate(initial_solution, rvalues, rpositions, distmx)

    archive = [initial_solution]
    front = []
    log = []

    archive, front, dominated = _initialize_archive(
        initial_solution, neighborhood, archive, rvalues, rpositions, distmx
    )
    
    save_statistics(log, front)

    start_time = time.perf_counter()
    
    archive, front, log = _run_main_optimization_loop(
        archive, front, dominated, log, neighborhood, 
        start_time, total_time, max_iterations, rvalues, rpositions, distmx
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Finished running in {elapsed_time:.2f} seconds.")

    return archive, front, log


def _initialize_archive(
    solution: Solution, 
    neighborhood: Neighborhood, 
    archive: list,
    rvalues: np.ndarray, 
    rpositions: np.ndarray, 
    distmx: np.ndarray
) -> tuple[list, list, list]:
    """Initialize the archive by exploring neighborhoods of the initial solution."""
    for neighborhood_id in range(neighborhood.num_neighborhoods):
        neighbors = local_search(
            solution, neighborhood, neighborhood_id, rvalues, rpositions, distmx
        )
        archive, front, dominated = update_archive(archive, neighbors, ARCHIVE_MAX_SIZE)
    
    return archive, front, dominated


def _run_main_optimization_loop(
    archive: list, 
    front: list, 
    dominated: list, 
    log: list,
    neighborhood: Neighborhood,
    start_time: float,
    total_time: int,
    max_iterations: int,
    rvalues: np.ndarray,
    rpositions: np.ndarray,
    distmx: np.ndarray
) -> tuple[list, list, list]:
    """Run the main optimization loop."""
    progress_bar = tqdm(total=max_iterations, desc="Progress", unit="it", dynamic_ncols=True)
    iteration = 0

    while iteration < max_iterations and time.perf_counter() - start_time < total_time:
        solution = select_solution_from_front(front, dominated, iteration)
        
        for neighborhood_id in range(neighborhood.num_neighborhoods):
            if time.perf_counter() - start_time > total_time:
                break

            perturbed_solution = perturb_solution(
                solution, neighborhood, rvalues, rpositions, distmx
            )
            neighbors = local_search(
                perturbed_solution, neighborhood, neighborhood_id, rvalues, rpositions, distmx
            )

            archive, front, dominated = update_archive(
                archive, [perturbed_solution] + neighbors, ARCHIVE_MAX_SIZE
            )

            save_statistics(log, front)
            
        iteration += 1
        progress_bar.update(1)

    progress_bar.close()
    return archive, front, log


def save_results_to_files(
    paths: list, 
    scores: np.ndarray, 
    log: list,
    output_dir: str, 
    map_name: str, 
    max_speed: float
) -> None:
    """
    Save optimization results to pickle files.
    
    Args:
        paths: List of solution paths
        scores: Array of solution scores
        log: Optimization log
        output_dir: Base output directory
        map_name: Name of the map used
        max_speed: Maximum speed value
    """
    results_dir = Path(output_dir) / map_name / str(max_speed)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for path, score in zip(paths, scores):
        with open(results_dir / "scores.pkl", "ab") as f:
            pickle.dump(score, f)
        with open(results_dir / "paths.pkl", "ab") as f:
            pickle.dump(path, f)
    
    with open(results_dir / "log.pkl", "ab") as f:
        pickle.dump(log, f)


def plot_solution_paths(
    paths: list,
    scores: np.ndarray,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    map_name: str,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> None:
    """
    Plot and save visualization of solution paths.
    
    Args:
        paths: List of solution paths
        scores: Array of solution scores
        rpositions: Positions of reward nodes
        rvalues: Reward values
        map_name: Name of the map
        output_dir: Output directory for plots
    """
    print("Plotting paths...")
    
    plots_dir = Path(output_dir) / DEFAULT_PATHS_SUBDIR / map_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    existing_files = len([f for f in plots_dir.iterdir() if f.is_file()])
    
    for i, (path, score) in enumerate(zip(paths, scores)):
        filename = str(existing_files + i + 1)
        plot.plot_paths_with_rewards(
            rpositions,
            rvalues,
            path,
            score,
            directory=str(plots_dir),
            fname=filename,
        )


def run_optimization(
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    budget: list[int],
    map_name: str,
    output_dir: str,
    begin: int = -1,
    end: int = -2,
    total_time: int = 600,
    num_agents: int = 1,
    speeds: list = [1],
    seed: int = 42,
    max_iterations: int = 100,
    algorithm: str = "unique_vis",
    save_results: bool = True,
    plot_results: bool = True,
) -> list:
    """
    Run the complete multi-objective optimization pipeline.
    
    Args:
        rpositions: Coordinates of reward nodes
        rvalues: Reward values for each node
        budget: Budget constraints for each agent
        map_name: Name of the map being used
        output_dir: Directory to save results
        begin: Index of the starting node
        end: Index of the ending node
        total_time: Maximum execution time in seconds
        num_agents: Number of agents in the system
        speeds: Speed values for each agent
        seed: Random seed for reproducibility
        max_iterations: Maximum number of iterations
        algorithm: Algorithm variant to use
        save_results: Whether to save results to files
        plot_results: Whether to generate plots
        
    Returns:
        List of Pareto optimal solution paths
    """
    Solution.set_parameters(begin, end, num_agents, budget, speeds)

    # Calculate distance matrix between all reward positions
    distance_matrix = cdist(rpositions, rpositions, metric="euclidean")

    archive, front, log = run_movns(
        rvalues, rpositions, distance_matrix, total_time, seed, max_iterations, algorithm
    )
    
    archive.sort(key=lambda solution: solution.score[0])

    paths = [solution.get_solution_paths() for solution in front]
    scores = np.array([solution.score for solution in front])

    if scores.size > 0:
        print(f"Best reward score: {max(scores[:, 0]):.2f}")
    else:
        print("No solutions found in Pareto front")

    if save_results and paths:
        save_results_to_files(paths, scores, log, output_dir, map_name, max(speeds))

    if plot_results and paths:
        plot_solution_paths(paths, scores, rpositions, rvalues, map_name)

    return paths
