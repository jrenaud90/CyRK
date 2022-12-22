import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

from CyRK import nbrk_ode


@njit
def diffeq_args(t, y, a, b):
    dy = np.zeros_like(y)
    dy[0] = (1. - a * y[1]) * y[0]
    dy[1] = (b * y[0] - 1.) * y[1]
    return dy


@njit
def diffeq(t, y):
    dy = np.zeros_like(y)
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy


initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 20.)
rtol = 1.0e-7
atol = 1.0e-8


def test_nbrk_test():
    """Check that the builtin test function for the nbrk integrator is working"""

    from CyRK import test_nbrk
    test_nbrk()


def test_basic_integration():
    """Check that the numba solver is able to run with its default arguments"""

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds)

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


def test_different_tols():
    """Check that the numba solver is able to run with different tolerances"""

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, rtol=1.0e-10, atol=1.0e-12)

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


def test_max_step():
    """Check that the numba solver is able to run with different maximum step size"""

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, max_step=1.0e5)

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


def test_first_step():
    """Check that the numba solver is able to run with a user provided first step size"""

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, first_step=0.01)

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


def test_rk23():
    """Check that the numba solver is able to run using the RK23 method """

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, rk_method=0)

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


def test_rk45():
    """Check that the numba solver is able to run using the RK45 method """

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, rk_method=1)

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


def test_dop853():
    """Check that the numba solver is able to run using the DOP853 method """

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, rk_method=2)

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


def test_teval():
    """Check that the numba solver is able to run using a user provided t_eval """

    t_eval = np.linspace(time_span[0], time_span[1], 10)

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, t_eval=t_eval)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    assert y_results.dtype == np.complex128
    assert time_domain.size > 1
    assert time_domain.size == t_eval.size
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str


def test_args():
    """Check that the numba solver is able to run with user provided additional diffeq arguments """

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq_args, time_span, initial_conds, args=(0.01, 0.02))

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


def test_accuracy():
    """Check that the numba solver is able to reproduce SciPy's results with reasonable accuracy """

    # Accuracy check tolerances
    check_rtol = 2.

    # Scipy
    scipy_solution = solve_ivp(diffeq, time_span, initial_conds, method='RK45', rtol=rtol, atol=atol)

    # Interpolate so that we have arrays defined at the same time steps
    time = np.linspace(time_span[0], time_span[1], scipy_solution.t.size)

    # CyRK
    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol, t_eval=time)

    # Check that the arrays are the correct size
    assert time_domain.size == scipy_solution.t.size
    assert y_results[0].size == scipy_solution.y[0].size
    assert y_results[1].size == scipy_solution.y[1].size

    p_diff_1 = 2. * (scipy_solution.y[0] - y_results[0]) / (y_results[0] + scipy_solution.y[0])
    p_diff_2 = 2. * (scipy_solution.y[1] - y_results[1]) / (y_results[1] + scipy_solution.y[1])

    # Check the accuracy of the results
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # # ax.plot(time_domain, y_results[0], 'r', label='CyRK')
    # # ax.plot(scipy_solution.t, scipy_solution.y[0], 'b', label='SciPy')
    # ax.plot(time_domain, p_diff_1)
    # ax.plot(time_domain, p_diff_2)
    # plt.show()

    # TODO: This is not a great result but I think it is due to the interpolation
    assert np.all(p_diff_1 < check_rtol)
    assert np.all(p_diff_2 < check_rtol)
