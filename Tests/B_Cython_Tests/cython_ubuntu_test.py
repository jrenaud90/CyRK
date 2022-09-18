import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

from CyRK import cyrk_ode


@njit
def diffeq(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]


initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8


def test_rk45():
    """Check that the cython solver is able to run using the RK45 method """

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span, initial_conds, rk_method=1)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    assert y_results.dtype == np.complex128
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str
