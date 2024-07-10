import numpy as np
from numba import njit

initial_conds = np.asarray((20., 20.), dtype=np.complex128)
initial_conds_float = np.asarray((20., 20.), dtype=np.float64)
time_span = (0., 20.)
rtol = 1.0e-7
atol = 1.0e-8


@njit
def cy_diffeq(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]

@njit
def pysolve_diffeq(dy, t, y):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]

@njit
def nb_diffeq(t, y):
    dy = np.empty_like(y)
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy


def test_cyrk():

    from CyRK import cyrk_ode

    time_domain, y_results, success, message = cyrk_ode(cy_diffeq, time_span, initial_conds)

    assert success
    assert type(time_domain) == np.ndarray
    assert type(y_results) == np.ndarray
    assert y_results.shape[0] == 2

    print("CyRK's cyrk_ode was tested successfully.")

def test_pysolver():

    from CyRK import pysolve_ivp

    result = pysolve_ivp(pysolve_diffeq, time_span, initial_conds_float, pass_dy_as_arg=True)

    assert result.success
    assert type(result.t) == np.ndarray
    assert type(result.y) == np.ndarray
    assert result.y.shape[0] == 2

    print("CyRK's PySolver was tested successfully.")

def test_nbrk():

    from CyRK import nbrk_ode

    time_domain, y_results, success, message = nbrk_ode(nb_diffeq, time_span, initial_conds)

    assert success
    assert type(time_domain) == np.ndarray
    assert type(y_results) == np.ndarray
    assert y_results.shape[0] == 2

    print("CyRK's nbrk_ode was tested successfully.")


def test_cysolver():

    from CyRK.cy.cysolverNew_test import cytester

    # TODO: Currently CySolver only works with floats not complex
    result = cytester(0, time_span, initial_conds, None, 1, 0, 0, 2000, rtol, atol)

    assert result.success
    assert type(result.t) == np.ndarray
    assert type(result.y) == np.ndarray
    assert result.y.shape[0] == 2

    print("CyRK's CySolver was tested successfully.")
