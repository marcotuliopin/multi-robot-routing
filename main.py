from utils import interpolate_paths
import plot
import numpy as np
import argparse
import ga.main as ga
import vns

MAX_DISTANCE_BETWEEN_AGENTS = 3
BUDGET = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-objective GA.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--plot-path", action="store_true", help="Plot the resulting best path.")
    parser.add_argument("--plot-distances", action="store_true", help="Plot the resulting best path.")
    parser.add_argument("--plot-interpolation", action="store_true", help="Plot the interpolated path.")
    parser.add_argument("--save-plot", type=str, default=None, help="Save the plot to the specified directory.")
    parser.add_argument("--run-animation", action="store_true", help="Run the animation of the best path.")
    parser.add_argument("--map", type=str, default="maps/grid_asymetric.txt", help="Path to the map image.")
    parser.add_argument("--method", type=str, default="nsga2", help="Optimization method.")
    args = parser.parse_args()
    
    # Read the rewards from the map file
    with open(args.map, "r") as f:
        lines = f.readlines()
        num_rewards = int(lines[0])
        rpositions = np.array(
            [list(map(float, line.split())) for line in lines[1 : num_rewards + 1]]
        )
        rvalues = np.array([float(line) for line in lines[num_rewards + 1 :]])

    # Run the GA
    match args.method:
        case "nsga2":
            path1, path2, logbook = ga.main(
                num_rewards,
                rpositions,
                rvalues,
                MAX_DISTANCE_BETWEEN_AGENTS,
                BUDGET,
                seed=42,
            )
        case "vns":
            path1, path2, logbook = vns.main(
                num_rewards,
                rpositions,
                rvalues,
                MAX_DISTANCE_BETWEEN_AGENTS,
                BUDGET,
                seed=42,
            )
        case _:
            raise ValueError(f"Invalid method: {args.method}")
        
    if args.save_plot:
        directory = f"imgs/{args.method}/{args.save_plot}"
    else: 
        directory = None

    if args.plot_path:
        plot.plot_paths_with_rewards(
            rpositions,
            rvalues,
            [path1, path2],
            MAX_DISTANCE_BETWEEN_AGENTS,
            directory=directory
        )

    if args.plot_interpolation:
        interpolated_paths = interpolate_paths(path1, path2, rpositions, 1)
        plot.plot_interpolated_individual(
            interpolate_paths(path1, path2, rpositions, 1),
            MAX_DISTANCE_BETWEEN_AGENTS,
            directory=directory
        )

    if args.plot_distances:
        plot.plot_distances(
            path1,
            path2,
            rpositions,
            MAX_DISTANCE_BETWEEN_AGENTS,
            1,
            directory=directory
        )
