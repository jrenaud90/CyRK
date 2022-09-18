import numpy as np
from numba import njit

initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 20.)
rtol = 1.0e-7
atol = 1.0e-8

@njit
def diffeq(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
