import pytest
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


initial_conds = np.asarray((20., 20.), dtype=np.float64)
initial_conds_complex = np.asarray((20. + 0.01j, 20. - 0.01j), dtype=np.complex128)
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8

rtols = np.asarray((1.0e-7, 1.0e-8), dtype=np.float64, order='C')
atols = np.asarray((1.0e-8, 1.0e-9), dtype=np.float64, order='C')


def test_nbrk_test():
    """Check that the builtin test function for the nbrk integrator is working"""

    from CyRK import test_nbrk
    test_nbrk()

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
@pytest.mark.parametrize('use_rtol_array', (True, False))
@pytest.mark.parametrize('use_atol_array', (True, False))
def test_basic_integration(use_atol_array, use_rtol_array, rk_method, complex_valued):
    """Check that the numba solver is able to run with its default arguments"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    tol_dict = dict()
    if use_atol_array:
        tol_dict['atols'] = atols
    else:
        tol_dict['atol'] = atol
    if use_rtol_array:
        tol_dict['rtols'] = rtols
    else:
        tol_dict['rtol'] = rtol

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, **tol_dict)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_different_tols(rk_method, complex_valued):
    """Check that the numba solver is able to run with different tolerances"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, rtol=1.0e-10, atol=1.0e-12)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_max_step(rk_method, complex_valued):
    """Check that the numba solver is able to run with different maximum step size"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, max_step=time_span[1] / 2.)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_first_step(rk_method, complex_valued):
    """Check that the numba solver is able to run with a user provided first step size"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, first_step=0.01)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_large_end_value(rk_method, complex_valued):
    """Check that the numba solver is able to run using a larger ending time value """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span_large, initial_conds_to_use, rk_method=rk_method)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_teval(rk_method, complex_valued):
    """Check that the numba solver is able to run using a user provided t_eval """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    t_eval = np.linspace(time_span[0], time_span[1], 10)

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, t_eval=t_eval)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == t_eval.size
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_args(rk_method, complex_valued):
    """Check that the numba solver is able to run with user provided additional diffeq arguments """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        nbrk_ode(diffeq_args, time_span, initial_conds_to_use, rk_method=rk_method, args=(0.01, 0.02))

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    print(message)
    assert type(success) == bool
    assert success
    assert type(message) == str

@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_accuracy(rk_method):
    """Check that the numba solver is able to reproduce a known functions integral with reasonable accuracy """

    # TODO: This is only checking one equation. Add other types of diffeqs to provide better coverage.

    # Differential Equation
    @njit
    def diffeq_accuracy(t, y):
        dy = np.empty_like(y)
        dy[0] = np.sin(t) - y[1]  # dydt = sin(t) - x(t)
        dy[1] = np.cos(t) + y[0]  # dxdt = cos(t) + y(t)
        return dy

    @njit
    def correct_answer(t, c1_, c2_):
        y = np.empty((2, t.size), dtype=np.float64)
        y[0] = -c1_ * np.sin(t) + c2_ * np.cos(t) - (np.cos(t) / 2)  # -c1 * sin(t) + c2 * cos(t) - cos(t) / 2
        # At t=0; y = c2 - 1/2
        y[1] = c2_ * np.sin(t) + c1_ * np.cos(t) + (np.sin(t) / 2)   # c2 * sin(t) + c1 * cos(t) + sin(t) / 2
        # At t=0; x = c1
        return y

    # Initial Conditions
    # y=0 --> c2 = + 1/2
    c2 = 0.5
    # x=1 --> c1 = + 1
    c1 = 1.0
    y0 = np.asarray((0., 1.), dtype=np.float64)
    time_span_ = (0., 10.)

    # CyRK.nbrk_ode
    time_domain, y_results, success, message = \
        nbrk_ode(diffeq_accuracy, time_span_, y0, rk_method=rk_method, rtol=rtol, atol=atol)
    real_answer = correct_answer(time_domain, c1, c2)

    if rk_method == 0:
        assert np.allclose(y_results, real_answer, rtol=1.0e-3, atol=1.0e-6)
    elif rk_method == 1:
        assert np.allclose(y_results, real_answer, rtol=1.0e-4, atol=1.0e-7)
    else:
        assert np.allclose(y_results, real_answer, rtol=1.0e-5, atol=1.0e-8)

    # Check the accuracy of the results
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(time_domain, y_results[0], 'r', label='CyRK')
    # ax.plot(time_domain, y_results[1], 'r:')
    # ax.plot(time_domain, real_answer[0], 'b', label='Analytic')
    # ax.plot(time_domain, real_answer[1], 'b:')
    # plt.show()

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_max_num_steps(rk_method, complex_valued):
    """Check that the numba solver is able to utilize the max_num_steps argument """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    # First test a number of max steps which is fine.
    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span_large, initial_conds_to_use, rk_method=rk_method, max_num_steps=1000000)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    if complex_valued:
        assert y_results.dtype == np.complex128
    else:
        assert y_results.dtype == np.float64
    assert time_domain.size > 1
    assert time_domain.size == y_results[0].size
    assert len(y_results.shape) == 2
    assert y_results[0].size == y_results[1].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str

    # Now test an insufficient number of steps
    time_domain, y_results, success, message = \
        nbrk_ode(diffeq, time_span_large, initial_conds_to_use, rk_method=rk_method, max_num_steps=4)

    # Check that the ndarrays make sense
    assert not success
    assert message == "Maximum number of steps (set by user) exceeded during integration."
