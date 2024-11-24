from deap import base, creator, tools
from sklearn.neighbors import KDTree
from rewards.grid2 import rpositions, rvalues
from scipy.spatial.distance import cdist
from utils import calculate_mutation_probability, get_last_valid_idx, interpolate_paths
import ga
import plot
import numpy as np
import random
import argparse

NUM_REWARDS = rpositions.shape[0]

PSIZE = 1200
NGEN = 200

CX = 0.9
MX = 0.7
MXDECAY = 0.5

MAXD = 3
BUDGET = 100

REINIT_RATE = 0.2


distmx = cdist(rpositions, rpositions, metric="euclidean")
kdtree = KDTree(rpositions)


creator.create("FitnessMulti", base.Fitness, weights=(10.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", ga.init_individual, creator.Individual, NUM_REWARDS, rpositions, MAXD, kdtree)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", ga.cx_individual)
toolbox.register("mutate", ga.mut_individual, indpb=0.3)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", ga.evaluate)


def main(toolbox, seed=None):
    random.seed(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("max_reward", lambda fits: max(fit[0] for fit in fits))
    stats.register("avg_reward", lambda fits: np.mean([fit[0] for fit in fits]))
    stats.register("min_distance", lambda fits: min(fit[1] for fit in fits))
    stats.register("avg_distance", lambda fits: np.mean([fit[1] for fit in fits]))

    logbook = tools.Logbook()
    
    population = toolbox.population(n=PSIZE)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, rvalues, rpositions, distmx, MAXD, BUDGET)
    
    population = toolbox.select(population, len(population))

    record = stats.compile(population)
    logbook.record(gen=0, **record)

    for gen in range(1, NGEN):
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        mutation_prob = calculate_mutation_probability(gen, NGEN, MX, MXDECAY)

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # Crossover
            if random.random() < CX:
                toolbox.mate(ind1, ind2)

            # Mutation
            if random.random() < mutation_prob:
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

            del ind1.fitness.values
            del ind2.fitness.values

        # Fitness evaluation
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind, rvalues, rpositions, distmx, MAXD, BUDGET)
        
        # Reinitialize worst individuals
        if gen % 10 == 0:
            worst_ind = tools.selWorst(offspring, int(len(offspring) * REINIT_RATE))
            for ind in worst_ind:
                ind[:] = toolbox.individual()
        
        # Selection NSGA-II
        population = toolbox.select(population + offspring, PSIZE)

        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

    return population, logbook

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-objective GA.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--plot-path", action="store_true", help="Plot the resulting best path.")
    parser.add_argument("--plot-distances", action="store_true", help="Plot the resulting best path.")
    parser.add_argument("--save-plot", type=str, default=None, help="Save the plot to a file.")
    parser.add_argument("--run-animation", action="store_true", help="Run the animation of the best path.")
    args = parser.parse_args()

    # Run the GA
    population, logbook = main(toolbox, seed=42)

    pareto_front = tools.emo.sortLogNondominated(population, len(population), first_front_only=True)
    individual = pareto_front[0]

    print(ga.evaluate(individual, rvalues, rpositions, distmx, MAXD, BUDGET))

    # Get the paths from the chosen individual
    last_idx1 = get_last_valid_idx(individual[0], distmx, BUDGET) + 1
    path1 = list(individual[0][:last_idx1])

    last_idx2 = get_last_valid_idx(individual[1], distmx, BUDGET) + 1
    path2 = list(individual[1][:last_idx2])

    # Set the beginning and end of the paths to the origin
    path1.append(0)
    path1.insert(0, 0)
    path2.append(0)
    path2.insert(0, 0)

    print(path1, path2)

    if args.plot_path:
        plot.plot_paths_with_rewards(rpositions, rvalues, [path1, path2], MAXD, args.save_plot)

        interpolated_paths = interpolate_paths(path1, path2, rpositions, 1)
        print('Norma')
        print(np.linalg.norm(interpolated_paths[0] - interpolated_paths[1], axis=1))
        print(np.any(np.linalg.norm(interpolated_paths[0] - interpolated_paths[1], axis=1) > MAXD))
        plot.plot_interpolated_individual(interpolate_paths(path1, path2, rpositions, 1), MAXD, save_plot=args.save_plot)

    if args.plot_distances:
        plot.plot_distances(path1, path2, rpositions, MAXD, 1, save_plot=args.save_plot)