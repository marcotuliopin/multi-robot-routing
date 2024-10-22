import numpy as np

# https://ieeexplore.ieee.org/document/1642245
def get_flow_velocity(point, vortices): # Code provided by Douglas G. Macharet.
    """
    Calculate the flow field at a given point based on the positions of the vortices.

    Parameters:
    point (numpy.ndarray): The point at which to calculate the flow field.
    vortices (list): A list of vortices, where each vortex is represented by its center, decay, and magnitude.

    Returns:
    numpy.ndarray: The flow field at the given point.
    """
    init = True
    for vortex in vortices:

        vortex_center, decay, magnitude = vortex
        distance = np.linalg.norm(point - vortex_center, axis=1)

        velocity_x = (
            -magnitude
            * ((point[:, 1] - vortex_center[1]) / (2 * np.pi * distance**2))
            * (1 - np.exp(-(distance**2 / decay**2)))
        )
        velocity_y = (
            magnitude
            * ((point[:, 0] - vortex_center[0]) / (2 * np.pi * distance**2))
            * (1 - np.exp(-(distance**2 / decay**2)))
        )
        Vaux = np.column_stack((velocity_x, velocity_y))

        if init:
            V = Vaux
            init = False
        else:
            V += Vaux
    return V