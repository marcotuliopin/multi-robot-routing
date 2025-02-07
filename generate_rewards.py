import numpy as np


def generate_dispersed_rewards(num_rewards, dispersion, output_file):
    # Gerar a primeira posição de recompensa aleatoriamente
    rpositions = [np.random.randint(0, dispersion, size=2)]

    # Gerar posições de recompensas subsequentes
    for _ in range(1, num_rewards):
        while True:
            new_pos = rpositions[-1] + np.random.randint(-dispersion, dispersion + 1, size=2)
            if np.all(new_pos >= 0) and np.all(new_pos < dispersion * num_rewards):
                rpositions.append(new_pos)
                break

    rpositions = np.array(rpositions)

    # Gerar valores de recompensas
    rvalues = np.random.randint(1, num_rewards * 2, size=num_rewards)

    # Salvar as recompensas em um arquivo
    with open(output_file, "w") as f:
        f.write(f"{num_rewards} {dispersion}\n")
        for pos in rpositions:
            f.write(f"{pos[0]} {pos[1]}\n")
        for value in rvalues:
            f.write(f"{value}\n")


if __name__ == "__main__":
    # args = [
    #     {
    #         "num_rewards": 10,
    #         "dispersion": 70.0,
    #         "output_file": "maps/dispersed_small.txt",
    #     },
    #     {
    #         "num_rewards": 30,
    #         "dispersion": 70.0,
    #         "output_file": "maps/dispersed_large.txt",
    #     },
    #     {
    #         "num_rewards": 50,
    #         "dispersion": 70.0,
    #         "output_file": "maps/dispersed_huge.txt",
    #     },
    #     {
    #         "num_rewards": 100,
    #         "dispersion": 70.0,
    #         "output_file": "maps/dispersed_gigantic.txt",
    #     },
    #     {
    #         "num_rewards": 200,
    #         "dispersion": 70.0,
    #         "output_file": "maps/dispersed_colossal.txt",
    #     },
    #     {
    #         "num_rewards": 500,
    #         "dispersion": 70.0,
    #         "output_file": "maps/dispersed_titanic.txt",
    #     },
    #     {
    #         "num_rewards": 1000,
    #         "dispersion": 70.0,
    #         "output_file": "maps/dispersed_humongous.txt",
    #     },
    # ]
    # for arg in args:
    #     generate_dispersed_rewards(
    #         arg["num_rewards"], arg["dispersion"], arg["output_file"]
    #     )
    generate_dispersed_rewards(100, 6, "maps/dispersed_near.txt")
