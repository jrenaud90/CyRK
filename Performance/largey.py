""" Large number of y Differential Equations """

import numpy as np
from numba import njit

from CyRK import nb2cy

num_y = 10_000

largey_y0 = 100.0 * np.ones(num_y, dtype=np.float64, order='C')
largey_args = (-0.5, 1.0e-5)
largey_time_span_1 = time_span = (0., 10.)
largey_time_span_2 = (0., 100.)


@njit(cache=True)
def largey_nb(t, y, decay_rate, forcing_scale):
    dy = np.empty_like(y)
    num_y = 10_000
    decay_rate_use = decay_rate
    
    # This diffeq converges so should be stable
    for i in range(num_y):
        if i % 1000 == 0:
            # Every 1000 make the decay rate a little worse so there is some difference in the y's
            decay_rate_use *= 0.95
        dy[i] = (decay_rate_use * y[i]) + i * forcing_scale
        # Add some coupling
        if i % 2 == 0 and (i + 1) < (num_y - 1):
            dy[i] += 0.5 * y[i + 1]
    return dy

largey_cy = nb2cy(largey_nb, use_njit=True, cache_njit=True)
