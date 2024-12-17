import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
from taylor_maccoll import taylor_maccoll


def conic_shock(mach_1: float, theta_s: float, gamma: float) -> tuple:
    """
    Calculates the conic shock relations for a given Mach number, wave angle angle, and ratio of specific heats.

    Args:
        mach_1 (float): Initial Mach number.
        theta_s (float): Flow deflection angle.
        gamma (float): Adiabatic index.

    Returns:
        tuple: A tuple containing the post-shock Mach number, the Mach number at the cone surface, the deflection angle in degrees, and the cone half angle in degrees.

    Raises:
        ValueError: If the input values are not valid.

    """
    # Oblique Shock Relations
    # NASA version for flow deflection angle
    tan_delta = 2 * (1/np.tan(theta_s)) * (((mach_1**2 * np.sin(theta_s)**2) - 1) / (mach_1**2 * (gamma + np.cos(2 * theta_s)) + 2))
    delta = np.arctan(tan_delta)

    # Anderson's version for post oblique shock mach number
    mach_n1 = mach_1 * np.sin(theta_s)
    mach_n2 = np.sqrt((1 + ((gamma-1)/2) * mach_n1**2) / (gamma*mach_n1**2 - ((gamma-1)/2)))
    mach_2 = mach_n2 / np.sin(theta_s - delta)
    m_theta = -mach_2 * np.sin(theta_s - delta)
    m_r = mach_2 * np.cos(theta_s - delta)

    def event_cone_surface(theta: float, v: list) -> float:
        """
        Checks if the event cone surface is reached.

        Args:
            theta (float): Integration angle.
            v (list): Velocity vector.

        Returns:
            float: The value of the event cone surface.

        """
        return v[1]

    event_cone_surface.terminal = True

    taylor_maccoll_wrapper = lambda theta, v: taylor_maccoll(theta, v, gamma)

    # Solve Conic Shock
    opts = {'rtol': 1e-13, 'atol': 1e-13}
    sol = solve_ivp(lambda t, y: taylor_maccoll_wrapper(t, y), [theta_s, 0], [m_r, m_theta], events=event_cone_surface, method='RK45', t_eval=np.linspace(theta_s, 0, 5000), **opts)

    cone_angle = np.degrees(sol.t[-1])
    m = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    mach_c = m[-1]
    delta = np.degrees(delta)

    return mach_2, mach_c, delta, cone_angle