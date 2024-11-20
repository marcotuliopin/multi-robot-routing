import numpy as np


grid_size = 5
spacing = 3

reward_p = np.array([
    [i * spacing, j * spacing] 
    for i in range(grid_size) 
    for j in range(grid_size)
])

reward_value = np.array(
    [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 37, 52, 40, 28]
)