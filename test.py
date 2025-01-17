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
# 5 vezes para 2 agentes e calcular a média e o desvio padrão do tempo de execução.

# for m in args["maps"]:
#     print(f"Executing map {m}...")

#     num_rewards, dispersion, rpositions, rvalues = read_map(m)
#     budget = dispersion * 10

#     exec_time = []
#     for i in range(5):
#         start = time.time()
#         paths = movns(
#             num_rewards,
#             rpositions,
#             rvalues,
#             budget,
#             seed=42,
#             num_agents=2,
#         )
#         end = time.time()
#         duration = end - start
#         exec_time.append(duration)
#         with open(f"tests/exec_time_{num_rewards}.txt", "a") as f:
#             f.write(str(duration))

#     exec_time_avg = np.mean(exec_time)
#     exec_time_std = np.std(exec_time)
#     exec_time = {
#         "avg": exec_time_avg,
#         "std": exec_time_std,
#     }
#     with open(f"tests/exec_time_{num_rewards}.pkl", "wb") as f:
#         pickle.dump(exec_time, f)

# Analisar o aumento do tempo de execução conforme o número de agentes aumenta. Para isso, executar o mapa de 50 recompensas
# 5 vezes para cada número de agentes e calcular a média e o desvio padrão do tempo de execução.

num_rewards, dispersion, rpositions, rvalues = read_map(args["maps"][0])
budget = dispersion * 10

for i in range(3):
    start = time.perf_counter()

    paths = movns(
        num_rewards,
        rpositions,
        rvalues,
        budget,
        seed=42,
        num_agents=6,
    )
    end = time.perf_counter()
    duration = end - start

    with open(f"tests/num_agents_{6}.txt", "a") as f:
        f.write(f"{str(duration)}\n")

for a in args["num_agents"][3:]:
    print(f"Executing with {a} agents map {args["maps"][0]}...")

    num_rewards, dispersion, rpositions, rvalues = read_map(args["maps"][0])
    budget = dispersion * 10

    exec_time = []
    for i in range(5):
        start = time.perf_counter()

        paths = movns(
            num_rewards,
            rpositions,
            rvalues,
            budget,
            seed=42,
            num_agents=a,
        )
        end = time.perf_counter()
        duration = end - start

        exec_time.append(duration)
        with open(f"tests/num_agents_{a}.txt", "a") as f:
            f.write(f"{str(duration)}\n")

# Analisar o aumento do tempo de execução conforme o budget aumenta.

# Analisar o número de vizinhos gerados conforme o número de agentes aumenta. Para isso, executar o mapa de 100 recompensas
# 10 vezes para cada número de agentes e calcular a média e o desvio padrão do número de vizinhos gerados em cada iteração.
