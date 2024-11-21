from deap import base, creator, tools
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from rewards.grid import rpositions, rvalues
from scipy.spatial.distance import cdist
from utils import add_neighbor_values, calculate_mutation_probability, disentangle_paths, get_last_valid_idx
import ga
import plots
import numpy as np
import random
import argparse

NUM_REWARDS = rpositions.shape[0]

PSIZE = 800
NGEN = 150

CX = 0.8
MX = 0.7
MXDECAY = 1.
TSIZE = 2

MAXD = 3.5
BUDGET = 40
PROXGAIN = 0.1

distmx = cdist(rpositions, rpositions, metric="euclidean")
kdtree = KDTree(rpositions)
# weights = add_neighbor_values(rvalues, rpositions, kdtree, MAXD)
weights = rvalues

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("genes", random.sample, range(NUM_REWARDS), NUM_REWARDS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MX)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", ga.evaluate, distmx=distmx, rpositions=rpositions, kdtree=kdtree, maxdist=MAXD, budget=BUDGET)


def main(toolbox, seed=None):
    random.seed(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    
    population = toolbox.population(n=PSIZE)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, weights=weights)
    
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
            ind.fitness.values = toolbox.evaluate(ind, weights=weights)
        
        # Selection NSGA-II
        population = toolbox.select(population + offspring, PSIZE)

        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

    return population, logbook


def create_second_agent(agent1: list, maxdist: float, kdtree: KDTree) -> list:
    agent2 = []

    for r1 in agent1:
        neighbors = kdtree.query_radius([rpositions[r1]], r=maxdist)[0]
        neighbors = neighbors[np.argsort(weights[neighbors])[::-1]]
        print(agent2)

        for neighbor in neighbors:
            if neighbor not in agent2 and neighbor not in agent1:
                agent2.append(neighbor)
                break

    return agent2

    
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

    agent1 = pareto_front[0]
    agent1 = agent1[:get_last_valid_idx(agent1, distmx, BUDGET)+1]
    print(agent1)

    agent2 = create_second_agent(agent1, MAXD, kdtree)

    if args.plot_path:
        fig, ax = plt.subplots(figsize=(7, 5))
        plots.plot_rewards(ax, rpositions, weights)
        plots.plot_path(ax, rpositions, agent1, distmx, BUDGET, MAXD, color='orange', show_radius=True)
        plots.plot_path(ax, rpositions, agent2, distmx, BUDGET, MAXD, color='green')

        ax.set_title("Path of the best individual")
        plt.grid(True)
        plt.axis('equal')
        plt.ylim(0, None)
        plt.xlim(0, None)

        if args.save_plot:
            plt.savefig(f'{args.save_plot}_pathe.png')

        plt.show()

    agent1, agent2 = disentangle_paths(agent1, agent2, rpositions)

    if args.plot_path:
        fig, ax = plt.subplots(figsize=(7, 5))
        plots.plot_rewards(ax, rpositions, weights)
        plots.plot_path(ax, rpositions, agent1, distmx, BUDGET, MAXD, color='orange', show_radius=True)
        plots.plot_path(ax, rpositions, agent2, distmx, BUDGET, MAXD, color='green')

        ax.set_title("Path of the best individual")
        plt.grid(True)
        plt.axis('equal')
        plt.ylim(0, None)
        plt.xlim(0, None)

        if args.save_plot:
            plt.savefig(f'{args.save_plot}_path.png')

        plt.show()

    if args.plot_distances:
        plots.plot_distances(agent1, agent2, distmx, BUDGET, args.save_plot)