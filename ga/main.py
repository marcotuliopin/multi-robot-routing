import random
from typing import List, Tuple
import numpy as np
from .utils import get_last_valid_idx
import ga.operators as operators
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from deap import tools, creator, base

PSIZE = 1200
NGEN = 250

CX = 0.9
MX = 0.3

REINIT_RATE = 0.2
REINIT_GEN = 20


def create_toolbox(num_rewards: int, kdtree: KDTree, rpositions: np.ndarray, max_distance_between_agents: float):
    creator.create("FitnessMulti", base.Fitness, weights=(10.0, -1.0))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        operators.init_individual,
        creator.Individual,
        num_rewards,
        rpositions,
        max_distance_between_agents,
        kdtree,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", operators.cx_individual)
    toolbox.register("mutate", operators.mut_individual, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", operators.evaluate)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("max_reward", lambda fits: max(fit[0] for fit in fits))
    stats.register("avg_reward", lambda fits: np.mean([fit[0] for fit in fits]))
    stats.register("min_distance", lambda fits: min(fit[1] for fit in fits))
    stats.register("avg_distance", lambda fits: np.mean([fit[1] for fit in fits]))
    return toolbox, stats


def evolve(
    num_rewards: np.ndarray,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    distmx: np.ndarray,
    kdtree: KDTree,
    max_distance_between_agents: float,
    budget: int,
    seed=None,
) -> Tuple[List, dict]:
    random.seed(seed)
    toolbox, stats = create_toolbox(num_rewards, kdtree, rpositions, max_distance_between_agents)
    logbook = tools.Logbook()

    population = toolbox.population(n=PSIZE)

    for ind in population:
        ind.fitness.values = toolbox.evaluate(
            ind, rvalues, rpositions, distmx, max_distance_between_agents, budget
        )

    record = stats.compile(population)
    logbook.record(gen=0, **record)

    population = toolbox.select(population, len(population))
    for gen in range(1, NGEN):
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Reinitialize worst individuals
        if gen % REINIT_GEN == 0:
            worst_ind = tools.selWorst(offspring, int(len(offspring) * REINIT_RATE))
            for ind in worst_ind:
                ind[:] = toolbox.individual()

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # Crossover
            if random.random() < CX:
                toolbox.mate(ind1, ind2)

            # Mutation
            if random.random() < MX:
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

            del ind1.fitness.values
            del ind2.fitness.values

        # Fitness evaluation
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(
                ind, rvalues, rpositions, distmx, max_distance_between_agents, budget
            )

        # Selection NSGA-II
        population = toolbox.select(population + offspring, PSIZE)

        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

    return population, logbook


def main(
    num_rewards: int,
    rpositions: np.ndarray,
    rvalues: np.ndarray,
    max_distance_between_agents: float,
    budget: int,
    seed=None,
):
    distmx = cdist(rpositions, rpositions, metric="euclidean")
    kdtree = KDTree(rpositions)
    population, logbook = evolve(
        num_rewards,
        rpositions,
        rvalues,
        distmx,
        kdtree,
        max_distance_between_agents,
        budget,
        seed,
    )

    # Get the first individual from the best pareto front
    pareto_front = tools.emo.sortLogNondominated(
        population, len(population), first_front_only=True
    )
    individual = pareto_front[0]

    # Get the paths from the chosen individual
    last_idx1 = get_last_valid_idx(individual[0], distmx, budget) + 1
    path1 = list(individual[0][:last_idx1])

    last_idx2 = get_last_valid_idx(individual[1], distmx, budget) + 1
    path2 = list(individual[1][:last_idx2])

    # Set the beginning and end of the paths to the origin
    path1.append(0)
    path1.insert(0, 0)
    path2.append(0)
    path2.insert(0, 0)

    return path1, path2, logbook
