import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List

from src.movns import run_optimization


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run multi-objective optimization for multi-agent routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--map", 
        type=str, 
        default="maps/grid_asymetric.txt",
        help="Path to the map file containing problem instance"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="out/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="unique_vis", 
        choices=["unique_vis", "multi_vis"],
        help="Algorithm variant to use"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--total_time", 
        type=int, 
        default=540,
        help="Maximum execution time in seconds"
    )
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=100,
        help="Maximum number of iterations"
    )
    
    parser.add_argument(
        "--speeds", 
        type=float, 
        nargs="+",
        help="Speed values for agents (overrides map file)"
    )
    parser.add_argument(
        "--budget", 
        type=float, 
        nargs="+",
        help="Budget values for agents (overrides map file)"
    )
    
    parser.add_argument(
        "--no_save", 
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--no_plot", 
        action="store_true",
        help="Don't generate plots"
    )
    
    return parser.parse_args()


def load_problem_instance(map_file_path: str) -> Tuple[int, int, List[float], List[float], np.ndarray, np.ndarray]:
    """
    Load problem instance from map file.
    
    Args:
        map_file_path: Path to the map file
        
    Returns:
        Tuple of (num_rewards, num_agents, budget, speeds, rpositions, rvalues)
        
    Raises:
        FileNotFoundError: If map file doesn't exist
        ValueError: If map file format is invalid
    """
    map_path = Path(map_file_path)
    
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_file_path}")
    
    try:
        with open(map_path, "r") as f:
            lines = f.readlines()
            
        # Parse header information
        num_rewards = int(float(lines[0].split(sep=";")[1]))
        num_agents = int(lines[1].split(sep=";")[1])
        default_budget = float(lines[2].split(sep=";")[1])
        
        # Initialize default values
        budget = [default_budget] * num_agents
        speeds = [1.0] * num_agents
        
        reward_data = []
        for line in lines[3:]:
            parts = line.strip().split(";")
            if len(parts) >= 3:
                x, y, value = float(parts[0]), float(parts[1]), float(parts[2])
                reward_data.append([x, y, value])
        
        if not reward_data:
            raise ValueError("No reward data found in map file")
        
        reward_array = np.array(reward_data)
        rpositions = np.vstack([reward_array[1:, :2], reward_array[0:1, :2]])
        rvalues = np.concatenate([reward_array[1:, 2], reward_array[0:1, 2]])
        
        return num_rewards, num_agents, budget, speeds, rpositions, rvalues
        
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid map file format: {e}")


def validate_and_update_parameters(
    args: argparse.Namespace,
    num_agents: int,
    default_budget: List[float],
    default_speeds: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Validate and update agent parameters from command line arguments.
    
    Args:
        args: Command line arguments
        num_agents: Number of agents from map file
        default_budget: Default budget values
        default_speeds: Default speed values
        
    Returns:
        Tuple of (budget, speeds)
        
    Raises:
        ValueError: If parameter counts don't match number of agents
    """
    budget = default_budget.copy()
    speeds = default_speeds.copy()
    
    # Override with command line arguments if provided
    if args.budget is not None:
        if len(args.budget) != num_agents:
            raise ValueError(f"Budget count ({len(args.budget)}) must match number of agents ({num_agents})")
        budget = args.budget
        
    if args.speeds is not None:
        if len(args.speeds) != num_agents:
            raise ValueError(f"Speed count ({len(args.speeds)}) must match number of agents ({num_agents})")
        speeds = args.speeds
    
    return budget, speeds


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_command_line_arguments()
        
        # Load problem instance from map file
        print(f"Loading problem instance from: {args.map}")
        num_rewards, num_agents, default_budget, default_speeds, rpositions, rvalues = (
            load_problem_instance(args.map)
        )
        
        print(f"Problem instance: {num_agents} agents, {num_rewards} rewards")
        
        # Validate and update parameters
        budget, speeds = validate_and_update_parameters(
            args, num_agents, default_budget, default_speeds
        )
        
        print(f"Agent configuration: speeds={speeds}, budget={budget}")
        
        # Extract map name for output organization
        map_name = Path(args.map).stem
        
        print(f"Starting optimization with algorithm: {args.algorithm}")
        paths = run_optimization(
            rpositions=rpositions,
            rvalues=rvalues,
            budget=budget,
            map_name=map_name,
            output_dir=args.out,
            total_time=args.total_time,
            num_agents=num_agents,
            speeds=speeds,
            seed=args.seed,
            max_iterations=args.num_iterations,
            algorithm=args.algorithm,
            save_results=not args.no_save,
            plot_results=not args.no_plot,
        )
        
        print(f"Optimization completed. Found {len(paths)} Pareto optimal solutions.")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()