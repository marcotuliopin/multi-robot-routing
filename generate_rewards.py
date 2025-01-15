import numpy as np
import argparse


def generate_dispersed_rewards(num_rewards, dispersion, output_file):
    # Gerar posições de recompensas dispersas
    rpositions = np.random.randint(0, dispersion, size=(num_rewards, 2))

    # Gerar valores de recompensas
    rvalues = np.arange(0, num_rewards * 5, 5)

    # Salvar as recompensas em um arquivo
    with open(output_file, "w") as f:
        f.write(f"{num_rewards} {dispersion}\n")
        for pos in rpositions:
            f.write(f"{pos[0]} {pos[1]}\n")
        for value in rvalues:
            f.write(f"{value}\n")


if __name__ == "__main__":
    args = [
        {
            "num_rewards": 10,
            "dispersion": 10.0,
            "output_file": "maps/dispersed_small.txt",
        },
        {
            "num_rewards": 30,
            "dispersion": 10.0,
            "output_file": "maps/dispersed_large.txt",
        },
        {
            "num_rewards": 50,
            "dispersion": 15.0,
            "output_file": "maps/dispersed_huge.txt",
        },
        {
            "num_rewards": 100,
            "dispersion": 20.0,
            "output_file": "maps/dispersed_gigantic.txt",
        },
        {
            "num_rewards": 200,
            "dispersion": 30.0,
            "output_file": "maps/dispersed_colossal.txt",
        },
        {
            "num_rewards": 500,
            "dispersion": 50.0,
            "output_file": "maps/dispersed_titanic.txt",
        },
        {
            "num_rewards": 1000,
            "dispersion": 70.0,
            "output_file": "maps/dispersed_humongous.txt",
        },
    ]
    for arg in args:
        generate_dispersed_rewards(
            arg["num_rewards"], arg["dispersion"], arg["output_file"]
        )
