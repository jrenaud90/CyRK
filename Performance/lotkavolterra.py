""" Lokavolterra Differential Equations """
import numpy as np
from numba import njit

from CyRK import nb2cy

lotkavolterra_y0 = np.asarray((10., 5.), dtype=np.float64)
lotkavolterra_args = (1.5, 1, 3, 1)
lotkavolterra_time_span_1 = (0., 15.)
lotkavolterra_time_span_2 = (0., 150.)


@njit(cache=True)
def lotkavolterra_nb(t, y, a, b, c, d):

    y0 = y[0]
    y1 = y[1]
    dy = np.empty_like(y)
    dy[0] = a * y0 - b * y0 * y1
    dy[1] = -c * y1 + d * y0 * y1
    return dy


lotkavolterra_cy = nb2cy(lotkavolterra_nb, use_njit=True, cache_njit=True)
