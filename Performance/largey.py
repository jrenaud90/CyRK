""" Large number of y Differential Equations """

import numpy as np
from numba import njit

from CyRK import nb2cy

num_y = 10_000

largey_y0 = 100.0 * np.ones(num_y, dtype=np.float64, order='C')
largey_args = (-0.5,)
largey_time_span_1 = time_span = (0., 10.)
largey_time_span_2 = (0., 100.)


@njit(cache=True)
def largey_nb(t, y, decay_rate):
    dy = np.empty_like(y)
    num_y = 10_000
    decay_rate_use = decay_rate
    
    # This diffeq converges so should be stable
    for i in range(num_y):
        decay_rate_use *= 0.9999
        if i < (num_y - 1):
            dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0 + y[i + 1]/50.0)
        else:
            dy[i] = decay_rate_use * y[i] * np.sin(2 * np.pi * t / 5.0)
    return dy

largey_cy = nb2cy(largey_nb, use_njit=True, cache_njit=True)


largey_simple_y0 = 100.0 * np.ones(num_y, dtype=np.float64, order='C')
largey_simple_args = (-0.5,)
largey_simple_time_span_1 = time_span = (0., 10.)
largey_simple_time_span_2 = (0., 100.)


@njit(cache=True)
def largey_simple_nb(t, y, decay_rate):
    dy = np.zeros_like(y)
    dy[0] = np.sin(2.0 * np.pi * t / 10.0)
    return dy

largey_simple_cy = nb2cy(largey_simple_nb, use_njit=True, cache_njit=True)

