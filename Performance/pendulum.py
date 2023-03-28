""" Pendulum Differential Equations """

import numpy as np
from numba import njit

from CyRK import nb2cy

pendulum_y0 = np.asarray((0.01, 0.), dtype=np.float64)
pendulum_args = (1., 1., 9.81)  # length [m], mass [kg], acceleration due to gravity [m s-2]
pendulum_time_span_1 = (0., 10.)
pendulum_time_span_2 = (0., 100.)


@njit(cache=True)
def pendulum_nb(t, y, l, m, g):

    # External torque
    torque = 0.1 * np.sin(t)

    y0 = y[0]  # Angular deflection [rad]
    y1 = y[1]  # Angular velocity [rad s-1]
    dy = np.empty_like(y)
    dy[0] = y1
    dy[1] = (-3. * g / (2. * l)) * np.sin(y0) + (3. / (m * l**2)) * torque
    return dy


pendulum_cy = nb2cy(pendulum_nb, use_njit=True, cache_njit=True)
