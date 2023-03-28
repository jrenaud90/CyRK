""" Lorenz Differential Equations """

import numpy as np
from numba import njit

from CyRK import nb2cy

lorenz_y0 = np.asarray((1., 0., 0.), dtype=np.float64)
lorenz_args = (10., 28.0, 8. / 3.)
lorenz_time_span_1 = (0., 10.)
lorenz_time_span_2 = (0., 100.)


@njit(cache=True)
def lorenz_nb(t, y, a, b, c):

    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    dy = np.empty_like(y)
    dy[0] = a * (y1 - y0)
    dy[1] = y0 * (b - y2) - y1
    dy[2] = y0 * y1 - c * y2
    return dy


lorenz_cy = nb2cy(lorenz_nb, use_njit=True, cache_njit=True)

@njit(cache=True)
def lorenz_nb_extra(t, y, a, b, c):

    y0 = y[0]
    y1 = y[1]
    y2 = y[2]

    e_1 = a
    e_2 = (b - y2)
    e_3 = c * y2

    dy_0 = e_1 * (y1 - y0)
    dy_1 = y0 * e_2 - y1
    dy_2 = y0 * y1 - e_3
    return np.asarray([dy_0, dy_1, dy_2, e_1, e_2, e_3], dtype=np.float64)


lorenz_cy_extra = nb2cy(lorenz_nb_extra, use_njit=True, cache_njit=True)
