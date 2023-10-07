import pytest
import numpy as np
from numba import njit

from CyRK import cyrk_ode
from CyRK.cy.cysolvertest import CySolverTester, CySolverAccuracyTest


@njit
def diffeq(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]


@njit
def diffeq_args(t, y, dy, a, b):
    dy[0] = (1. - a * y[1]) * y[0]
    dy[1] = (b * y[0] - 1.) * y[1]

initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
initial_conds_complex = np.asarray((20. + 0.01j, 20. - 0.01j), dtype=np.complex128, order='C')
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8

rtols = np.asarray((1.0e-7, 1.0e-8), dtype=np.float64, order='C')
atols = np.asarray((1.0e-8, 1.0e-9), dtype=np.float64, order='C')

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
@pytest.mark.parametrize('use_rtol_array', (True, False))
@pytest.mark.parametrize('use_atol_array', (True, False))
def test_basic_integration_cyrk_ode(use_atol_array, use_rtol_array, rk_method, complex_valued):
    """Check that the cython function solver is able to run with its default arguments"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    if use_atol_array:
        atols_use = atols
    else:
        atols_use = None
    if use_rtol_array:
        rtols_use = rtols
    else:
        rtols_use = None

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span, initial_conds_to_use,
                 rk_method=rk_method, rtol=rtol, atol=atol, rtols=rtols_use, atols=atols_use)

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

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
@pytest.mark.parametrize('use_rtol_array', (True, False))
@pytest.mark.parametrize('use_atol_array', (True, False))
def test_basic_integration_CySolverTester(use_atol_array, use_rtol_array, rk_method, complex_valued):
    """Check that the cython class solver is able to run with its default arguments"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    if use_atol_array:
        atols_use = atols
    else:
        atols_use = None
    if use_rtol_array:
        rtols_use = rtols
    else:
        rtols_use = None

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use,
                                        rtol=rtol, atol=atol, rtols=rtols_use, atols=atols_use,
                                        rk_method=rk_method, auto_solve=True)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str


@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_different_tols(rk_method, complex_valued):
    """Check that the cython function solver is able to run with different tolerances"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, rtol=1.0e-10, atol=1.0e-12)

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

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_different_tols_CySolverTester(rk_method, complex_valued):
    """Check that the cython class solver is able to run with different tolerances"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method,
                                        rtol=1.0e-10, atol=1.0e-12, auto_solve=True)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_max_step(rk_method, complex_valued):
    """Check that the cython function solver is able to run with different maximum step size"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, max_step=time_span[1] / 2.)

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

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_max_step_CySolverTester(rk_method, complex_valued):
    """Check that the cython class solver is able to run with different maximum step size"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method, max_step=time_span[1] / 2., auto_solve=True)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_first_step(rk_method, complex_valued):
    """Check that the cython function solver is able to run with a user provided first step size"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, first_step=0.01)

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

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_first_step_CySolverTester(rk_method, complex_valued):
    """Check that the cython class solver is able to run with a user provided first step size"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method, first_step=0.01, auto_solve=True)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_large_end_value(rk_method, complex_valued):
    """Check that the cython function solver is able to run using the DOP853 method. Using a larger ending time value """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span_large, initial_conds_to_use, rk_method=rk_method)

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

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_large_end_value_CySolverTester(rk_method, complex_valued):
    """Check that the cython class solver is able to run using the DOP853 method. Using a larger ending time value """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span_large, initial_conds_to_use, rk_method=rk_method, auto_solve=True)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_teval(rk_method, complex_valued):
    """Check that the cython function solver is able to run using a user provided t_eval """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    t_eval = np.linspace(time_span[0], time_span[1], 10)

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=rk_method, t_eval=t_eval)

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

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_teval_CySolverTester(rk_method, complex_valued):
    """Check that the cython class solver is able to run using a user provided t_eval """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    t_eval = np.linspace(time_span[0], time_span[1], 10)

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method, t_eval=t_eval, auto_solve=True)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_args(rk_method, complex_valued):
    """Check that the cython function solver is able to run with user provided additional diffeq arguments """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq_args, time_span, initial_conds_to_use, rk_method=rk_method, args=(0.01, 0.02))

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

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_args_CySolverTester(rk_method, complex_valued):
    """Check that the cython class solver is able to run with user provided additional diffeq arguments """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method, args=(0.01, 0.02), auto_solve=True)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str

@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_accuracy(rk_method):
    """Check that the cython function solver is able to reproduce a known functions integral with reasonable accuracy """

    # TODO: This is only checking one equation. Add other types of diffeqs to provide better coverage.

    # Differential Equation
    @njit
    def diffeq_accuracy(t, y, dy):
        dy[0] = np.sin(t) - y[1]  # dydt = sin(t) - x(t)
        dy[1] = np.cos(t) + y[0]  # dxdt = cos(t) + y(t)

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

    # CyRK.cyrk_ode
    time_domain, y_results, success, message = \
        cyrk_ode(diffeq_accuracy, time_span_, y0, rk_method=rk_method, rtol=1.0e-8, atol=1.0e-9)
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
    
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_accuracy_CySolverTester(rk_method):
    """Check that the cython class solver is able to reproduce a known functions integral with reasonable accuracy """

    # TODO: This is only checking one equation. Add other types of diffeqs to provide better coverage.

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

    # CyRK.CySolver
    CySolverAccuracyTestInst = CySolverAccuracyTest(time_span_, y0, rk_method=rk_method, rtol=1.0e-8, atol=1.0e-9, auto_solve=True)
    real_answer = correct_answer(CySolverAccuracyTestInst.t, c1, c2)

    if rk_method == 0:
        assert np.allclose(CySolverAccuracyTestInst.y, real_answer, rtol=1.0e-3, atol=1.0e-6)
    elif rk_method == 1:
        assert np.allclose(CySolverAccuracyTestInst.y, real_answer, rtol=1.0e-4, atol=1.0e-7)
    else:
        assert np.allclose(CySolverAccuracyTestInst.y, real_answer, rtol=1.0e-5, atol=1.0e-8)

    # Check the accuracy of the results
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(CySolverAccuracyTestInst.t, CySolverAccuracyTestInst.y[0], 'r', label='CyRK')
    # ax.plot(CySolverAccuracyTestInst.t, CySolverAccuracyTestInst.y[1], 'r:')
    # ax.plot(CySolverAccuracyTestInst.t, real_answer[0], 'b', label='Analytic')
    # ax.plot(CySolverAccuracyTestInst.t, real_answer[1], 'b:')
    # plt.show()

@pytest.mark.parametrize('complex_valued', (True, False))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_max_num_steps(rk_method, complex_valued):
    """Check that the cython function cyrk_ode can use max_num_steps argument """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    # First test a number of max steps which is fine.
    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span_large, initial_conds_to_use, rk_method=rk_method, max_num_steps=1000000)

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
        cyrk_ode(diffeq, time_span_large, initial_conds_to_use, rk_method=rk_method, max_num_steps=4)

    assert not success
    assert message == "Maximum number of steps (set by user) exceeded during integration."

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_max_num_steps_CySolverTester(rk_method, complex_valued):
    """Check that the cython class solver correctly uses the max_num_steps argument. """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    # First test a number of max steps which is fine.
    CySolverTesterInst = CySolverTester(time_span_large, initial_conds_to_use, rk_method=rk_method, auto_solve=True, max_num_steps=1000000)

    # Check that the ndarrays make sense
    assert type(CySolverTesterInst.t) == np.ndarray
    assert CySolverTesterInst.t.dtype == np.float64
    if complex_valued:
        assert CySolverTesterInst.y.dtype == np.complex128
    else:
        assert CySolverTesterInst.y.dtype == np.float64
    assert CySolverTesterInst.t.size > 1
    assert CySolverTesterInst.t.size == CySolverTesterInst.y[0].size
    assert len(CySolverTesterInst.y.shape) == 2
    assert CySolverTesterInst.y[0].size == CySolverTesterInst.y[1].size

    # Check that the other output makes sense
    assert type(CySolverTesterInst.success) == bool
    assert CySolverTesterInst.success
    assert type(CySolverTesterInst.message) == str

    # Now test an insufficient number of steps
    CySolverTesterInst = CySolverTester(
        time_span_large, initial_conds_to_use, rk_method=rk_method, auto_solve=True, max_num_steps=4)

    assert not CySolverTesterInst.success
    assert CySolverTesterInst.status == -2

def test_bad_tols_cyrk():

    # Too many rtols and atols
    bad_rtols = np.asarray((1.0e-6, 1.0e-7, 1.0e-8), dtype=np.float64, order='C')
    bad_atols = np.asarray((1.0e-7, 1.0e-8, 1.0e-9), dtype=np.float64, order='C')

    with pytest.raises(AttributeError):
        time_domain, y_results, success, message = \
            cyrk_ode(diffeq, time_span_large, initial_conds, rk_method=1, rtols=bad_rtols)

    with pytest.raises(AttributeError):
        time_domain, y_results, success, message = \
            cyrk_ode(diffeq, time_span_large, initial_conds, rk_method=1, atols=bad_atols)


def test_bad_tols_CySolver():

    # Too many rtols and atols
    bad_rtols = np.asarray((1.0e-6, 1.0e-7, 1.0e-8), dtype=np.float64, order='C')
    bad_atols = np.asarray((1.0e-7, 1.0e-8, 1.0e-9), dtype=np.float64, order='C')

    with pytest.raises(AttributeError):
        CySolverTesterInst = CySolverTester(time_span_large, initial_conds,
                                            rk_method=1, rtols=bad_rtols, auto_solve=True)

    with pytest.raises(AttributeError):
        CySolverTesterInst = CySolverTester(time_span_large, initial_conds,
                                            rk_method=1, atols=bad_atols, auto_solve=True)


if __name__ == '__main__':

    test_basic_integration_cyrk_ode(False, False, 2, False)
