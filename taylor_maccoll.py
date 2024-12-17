import numpy as np

# Taylor-Maccoll function
def taylor_maccoll(theta: float, v: list, gamma: float) -> list:
    """
    Calculates the derivatives of the Taylor-Maccoll function.

    Args:
        theta (float): Flow angle.
        v (list): Velocity vector.

    Returns:
        list: The derivatives of the Taylor-Maccoll function.

    """
    g = 1.4
    dydt = [v[1] + ((g-1)/2)* v[0]*v[1] * ((v[0] + v[1] * (1/np.tan(theta))) / (v[1]**2 - 1)),
            -v[0] + (1 + ((g-1)/2) * v[1]**2) * ((v[0] + v[1] * (1/np.tan(theta))) / (v[1]**2 - 1))]
    return dydt
