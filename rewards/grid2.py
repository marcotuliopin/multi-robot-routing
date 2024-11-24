import numpy as np

grid_size = 5
spacing = 3

rpositions = np.array([
    [i * spacing, j * spacing] 
    for i in range(grid_size) 
    for j in range(grid_size)
])

rvalues = np.array([
    90, 50, 100, 40, 85,  
    30, 15, 25, 20, 35,  
    10, 18, 22, 17, 12,  
    25, 28, 33, 26, 40,  
    88, 50, 42, 55, 95
])
