import random
from typing import Any, Callable

import numpy as np
from tqdm import tqdm
from using_clusters.rewards import rewards
from deap import base, creator, tools
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

N_REWARDS = rewards.shape[0]
POP_SIZE = 180
BUDGET = 20
N_GEN = 400
CX_PROB = 0.9
MUT_PROB = 0.3
MUT_DECAY_RATE = N_GEN / 100

MAX_DISTANCE = 4
MIN_SAMPLES = 2
dbscan = DBSCAN(eps=MAX_DISTANCE / 2, min_samples=MIN_SAMPLES)
clusters = dbscan.fit_predict(rewards[:, :2]) 

distance_mx = cdist(rewards[:, :2], rewards[:, :2], metric="euclidean")

def init_individual(icls: Any) -> list:
    genes = [[], []]
    clusters_to_visit = np.unique(clusters)

    while len(genes[0]) < rewards.shape[0] and len(genes[1]) < rewards.shape[0]:
        cluster = np.random.choice(clusters_to_visit)
        cluster_points = np.where(clusters == cluster)[0]

        not_visited0 = cluster_points[~np.isin(cluster_points, genes[0])]
        not_visited1 = cluster_points[~np.isin(cluster_points, genes[1])]

        if len(not_visited0) == 0 or len(not_visited1) == 0:
            clusters_to_visit = clusters_to_visit[clusters_to_visit != cluster]
            continue

        num_points_to_visit = 1 if len(not_visited0) == 1 else np.random.randint(1, len(not_visited0))

        genes[0].extend(np.random.choice(not_visited0, size=num_points_to_visit, replace=False))
        genes[1].extend(np.random.choice(not_visited1, size=num_points_to_visit, replace=False))

    return icls(np.array(genes))


def cx_partially_matched(ind1: np.ndarray, ind2: np.ndarray) -> tuple:
    size = min(len(ind1[0]), len(ind2[0]))
    p1, p2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int)

    for i in range(size):
        p1[ind1[0][i]] = i
        p2[ind2[0][i]] = i

    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    for i in range(cxpoint1, cxpoint2):
        temp1 = ind1[0][i]
        temp2 = ind2[0][i]
        ind1[0][i], ind1[0][p1[temp2]] = temp2, temp1
        ind2[0][i], ind2[0][p2[temp1]] = temp1, temp2
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


def mut_individual(individual: np.ndarray, indpb: float) -> tuple:
    if np.random.random() < 0.5:
        mut_shuffle_idx(individual, indpb) 
        return

    path1, path2 = individual
    if np.random.random() < 0.5:
        mut_shuffle_idx_intra_cluster(path1, indpb)
    else:
        mut_shuffle_idx_intra_cluster(path2, indpb)
    return (individual,)


def mut_shuffle_idx(individual: np.ndarray, indpb: float) -> np.ndarray:
    for i in range(individual.shape[1]):
        if np.random.random() < indpb:
            mx_2 = np.random.randint(0, individual.shape[1])
            individual[:, i], individual[:, mx_2] = individual[:, mx_2].copy(), individual[:, i].copy()

    return (individual,)


def mut_shuffle_idx_intra_cluster(path: np.ndarray, indpb: float) -> np.ndarray:
    for i in range(len(path)):
        if np.random.random() < indpb:
            cluster = clusters[path[i]]
            cluster_points = np.where(clusters == cluster)[0]
            if len(cluster_points) > 1:
                mx_2 = np.random.choice(cluster_points)
                path[i], path[mx_2] = path[mx_2], path[i]

    return (path,)


def calculate_mutation_probability(generation: int, max_generations: int, initial_prob: float, decay_rate: float) -> float:
    return initial_prob * np.exp(-decay_rate * (generation / max_generations))


def eval_individual(individual: np.ndarray, distance_mx: np.ndarray) -> tuple:
    fitness = 0
    visited = set()

    for path in individual:
        total_distance = 0

        idx = path[0]
        if idx not in visited:
            fitness += rewards[idx, 2]
            visited.add(idx)

        distance_to_first_point = np.linalg.norm(rewards[idx, :2])
        if (total_distance + distance_to_first_point) > BUDGET:
            break
        total_distance += distance_to_first_point

        for i in range(1, len(path)):
            idx = path[i]
            distance = distance_mx[path[i - 1], idx]
            if (total_distance + distance) > BUDGET:
                break
            total_distance += distance

            if idx not in visited:
                fitness += rewards[idx, 2]
                visited.add(idx)

    return (fitness,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", init_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cx_partially_matched)
toolbox.register("mutate", mut_individual, indpb=MUT_PROB)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_individual, distance_mx=distance_mx)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()


def evolve():
    population = toolbox.population(n=POP_SIZE)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    record = stats.compile(population)
    logbook.record(gen=0, **record)

    for gen in tqdm(range(N_GEN), desc='Progress'):
        elite = tools.selBest(population, 1)[0]
        offspring = toolbox.select(population, len(population) - 1)
        offspring = list(map(toolbox.clone, offspring))

        for _ in range(len(offspring) // 2):
            parent1, parent2 = random.sample(offspring, 2)
            if np.random.random() < CX_PROB:
                toolbox.mate(parent1, parent2)
                del parent1.fitness.values
                del parent2.fitness.values
            
        mutation_prob = calculate_mutation_probability(gen, N_GEN, MUT_PROB, MUT_DECAY_RATE)
        for individual in offspring:
            if np.random.random() < mutation_prob:
                toolbox.mutate(individual)
                del individual.fitness.values

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        offspring.append(elite)
        population[:] = offspring

        record = stats.compile(population)
        logbook.record(gen=0, **record)

    print(logbook)
    return tools.selBest(population, 1)[0]

ind = toolbox.individual()
toolbox.mutate(ind)