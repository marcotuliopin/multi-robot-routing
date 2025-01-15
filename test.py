# 5 mapas de tamanhos diferentes
# 20 recompensas, 100 recompensas, 200 recompensas, 500 recompensas, 1000 recompensas

# 10 números de agentes diferentes
# 2, 3, 6, 8, 10, 12, 14, 16, 18, 20


import pickle
import time
import numpy as np

from src import movns


args = {
    "maps": [
        "maps/dispersed_large.txt",
        "maps/dispersed_gigantic.txt",
        "maps/dispersed_colossal.txt",
        "maps/dispersed_titanic.txt",
        "maps/dispersed_humongous.txt",
    ],
    "num_agents": [2, 3, 6, 8, 10, 12, 14, 16, 18, 20],
}


def read_map(m):
    with open(m, "r") as f:
        lines = f.readlines()
        num_rewards, dispersion = lines[0].split()
        num_rewards = int(num_rewards)
        dispersion = float(dispersion)
        rpositions = np.array(
            [list(map(float, line.split())) for line in lines[1 : num_rewards + 1]]
        )
        rvalues = np.array([float(line) for line in lines[num_rewards + 1 :]])
    return num_rewards, dispersion, rpositions, rvalues


# Analisar o aumento do tempo de execução conforme o número de recompensas aumenta. Para isso, executar cada mapa
# 5 vezes para 3 agentes e calcular a média e o desvio padrão do tempo de execução.

map_exec_time = {}

for m in args["maps"]:
    print(f"Executing map {m}...")

    num_rewards, dispersion, rpositions, rvalues = read_map(m)
    budget = dispersion * 20

    exec_time = []
    for i in range(5):
        start = time.time()
        paths = movns(
            num_rewards,
            rpositions,
            rvalues,
            budget,
            seed=42,
            num_agents=3,
        )
        end = time.time()
        duration = end - start
        exec_time.append(duration)

    exec_time_avg = np.mean(exec_time)
    exec_time_std = np.std(exec_time)
    map_exec_time[num_rewards] = {
        "avg": exec_time_avg,
        "std": exec_time_std,
    }

with open("tests/map_exec_time.pkl", "wb") as f:
    pickle.dump(map_exec_time, f)

# Analisar o aumento do tempo de execução conforme o número de agentes aumenta. Para isso, executar o mapa de 100 recompensas
# 10 vezes para cada número de agentes e calcular a média e o desvio padrão do tempo de execução.

# Analisar o aumento do tempo de execução conforme o budget aumenta.

# Analisar o número de vizinhos gerados conforme o número de agentes aumenta. Para isso, executar o mapa de 100 recompensas
# 10 vezes para cada número de agentes e calcular a média e o desvio padrão do número de vizinhos gerados em cada iteração.
