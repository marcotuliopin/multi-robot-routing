from deap import base, creator, tools
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from rewards.grid import rpositions, rvalues
from scipy.spatial.distance import cdist
from utils import add_neighbor_values, calculate_mutation_probability
import ga
import plots
import numpy as np
import random
import argparse

NUM_REWARDS = rpositions.shape[0]

PSIZE = 800
NGEN = 500

CX = 0.8
MX = 0.7
MXDECAY = 1.
TSIZE = 4

MAXD = 3
BUDGET = 80

distmx = cdist(rpositions, rpositions, metric="euclidean")
kdtree = KDTree(rpositions)


creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", ga.init_individual, creator.Individual, NUM_REWARDS, rpositions, MAXD, kdtree)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", ga.cx_individual)
toolbox.register("mutate", ga.mut_individual, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", ga.evaluate)


def main(toolbox, seed=None):
    random.seed(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    
    nvalues = add_neighbor_values(rvalues, rpositions, kdtree, MAXD)    

    population = toolbox.population(n=PSIZE)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, 0, NGEN, nvalues, distmx, MAXD, BUDGET)
    
    population = toolbox.select(population, len(population))

    record = stats.compile(population)
    logbook.record(gen=0, **record)

    for gen in range(1, NGEN):
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX:
                toolbox.mate(parent1, parent2)
                del parent1.fitness.values
                del parent2.fitness.values
            
        # Mutation
        mutation_prob = calculate_mutation_probability(gen, NGEN, MX, MXDECAY)
        for individual in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(individual)
                del individual.fitness.values

        # Fitness evaluation
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind, gen, NGEN, nvalues, distmx, MAXD, BUDGET)
        
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

    if args.plot_path:
        fig, ax = plt.subplots(figsize=(7, 5))
        plots.plot_rewards(ax, rpositions, rvalues)
        plots.plot_path(ax, rpositions, individual[0], distmx, BUDGET, MAXD, color='orange')
        plots.plot_path(ax, rpositions, individual[1], distmx, BUDGET, MAXD, color='green')

        ax.set_title("Second Phase Individual Paths")
        plt.grid(True)
        plt.axis('equal')
        plt.ylim(0, None)
        plt.xlim(0, None)
        plt.show()
        if args.save_plot:
            plt.savefig(f'{args.save_plot}_path.png')

    if args.plot_distances:
        plots.plot_distances(individual[0], individual[1], distmx, BUDGET)
        if args.save_plot:
            plt.savefig(f'{args.save_plot}_distances.png')