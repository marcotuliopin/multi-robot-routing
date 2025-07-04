from src import movns
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-objective GA.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--map", type=str, default="maps/grid_asymetric.txt", help="Path to the map image.")
    parser.add_argument("--total_time", type=int, default=540, help="Execution time in seconds.")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--speeds", type=float, nargs="+", help="Speed of the agents.")
    parser.add_argument("--budget", type=float, nargs="+", help="Budget of the agents.")
    parser.add_argument("--algorithm", type=int, default=0, help="Indicates which algorithm to use. 0 or 1.")
    parser.add_argument("--out", type=str, default="out/")
    args = parser.parse_args()
    
    # Read the rewards from the map file
    with open(args.map, "r") as f:
        lines = f.readlines()
        num_rewards = float(lines[0].split(sep=";")[1])
        num_agents = int(lines[1].split(sep=";")[1])
        budget = [float(lines[2].split(sep=";")[1])] * int(num_agents)
        speeds = [1] * int(num_agents)

        rpositions = np.array(
            [list(map(float, line.split(sep=";")[:-1])) for line in lines[3:]]
        )
        rvalues = np.array([float(line.split(";")[2]) for line in lines[3:]])
        rpositions = np.append(rpositions[1:], [rpositions[0]], axis=0)
        rvalues = np.append(rvalues[1:], rvalues[0])
    
    if args.budget is not None:
        budget = args.budget
    if args.speeds is not None:
        speeds = args.speeds

    print(speeds)
    paths = movns(
        rpositions,
        rvalues,
        budget,
        map=args.map,
        out=args.out,
        seed=42,
        num_agents=num_agents,
        speeds=speeds,
        total_time=args.total_time,
        max_it=args.num_iterations,
        algorithm=args.algorithm,
    )