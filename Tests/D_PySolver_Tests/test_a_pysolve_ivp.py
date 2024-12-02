import numpy as np
import pytest
from numba import njit

from CyRK import pysolve_ivp, WrapCySolverResult


# To reduce number of tests, only test RK23 once since RK45 should capture all its functionality
SKIP_SOME_RK23_TESTS = True
RK23_TESTED = False

def diffeq(dy, t, y):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]


def diffeq_stiff(t, y):
    # TODO: Replace with function that is actually stiff
    dy = np.empty(y.size, dtype=np.float64)
    dy[0] = (1. - 0.01 * y[1]) * np.exp(y[0])
    dy[1] = (0.02 * y[0] - 1.) * np.exp(y[1])
    return dy

def diffeq_args(dy, t, y, a, b):
    dy[0] = (1. - a * y[1]) * y[0]
    dy[1] = (b * y[0] - 1.) * y[1]

def diffeq_scipy_style(t, y):
    dy = np.empty(y.size, dtype=np.float64)
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy

def diffeq_scipy_style_args(t, y, a, b):
    dy = np.empty(y.size, dtype=np.float64)
    dy[0] = (1. - a * y[1]) * y[0]
    dy[1] = (b * y[0] - 1.) * y[1]
    return dy

def diffeq_extra(dy, t, y):
    extra_1 = (1. - 0.01 * y[1])
    extra_2 = (0.02 * y[0] - 1.)
    dy[0] = extra_1 * y[0]
    dy[1] = extra_2 * y[1]
    dy[2] = extra_1
    dy[3] = extra_2

def diffeq_args_extra(dy, t, y, a, b):
    extra_1 = (1. - a * y[1])
    extra_2 = (b * y[0] - 1.)
    dy[0] = extra_1 * y[0]
    dy[1] = extra_2 * y[1]
    dy[2] = extra_1
    dy[3] = extra_2

def diffeq_scipy_style_extra(t, y):
    dy = np.empty(y.size + 2, dtype=np.float64)
    extra_1 = (1. - 0.01 * y[1])
    extra_2 = (0.02 * y[0] - 1.)
    dy[0] = extra_1 * y[0]
    dy[1] = extra_2 * y[1]
    dy[2] = extra_1
    dy[3] = extra_2
    return dy

def diffeq_scipy_style_args_extra(t, y, a, b):
    dy = np.empty(y.size + 2, dtype=np.float64)
    extra_1 = (1. - a * y[1])
    extra_2 = (b * y[0] - 1.)
    dy[0] = extra_1 * y[0]
    dy[1] = extra_2 * y[1]
    dy[2] = extra_1
    dy[3] = extra_2
    return dy

args = (0.01, 0.02)

initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-4
atol = 1.0e-7

rtols = np.asarray((1.0e-4, 1.0e-5), dtype=np.float64, order='C')
atols = np.asarray((1.0e-6, 1.0e-7), dtype=np.float64, order='C')


def test_pysolve_ivp_test():
    """Check that the builtin test function for the PySolver integrator is working"""

    from CyRK import test_pysolver
    test_pysolver()

# njit is slow during testing so only do it once for each diffeq
njit_rk23_tested = False
njit_rk45_tested = False
njit_DOP853_tested = False

@pytest.mark.filterwarnings("error")  # Some exceptions get propagated via cython as warnings; we want to make sure the lead to crashes.
@pytest.mark.parametrize('capture_extra', (True, False))
@pytest.mark.parametrize('max_step', (1.0, 100_000.0))
@pytest.mark.parametrize('first_step', (0.0, 0.00001))
@pytest.mark.parametrize('integration_method', ("RK23", "RK45", "DOP853"))
@pytest.mark.parametrize('use_different_tols', (True, False))
@pytest.mark.parametrize('use_rtol_array', (True, False))
@pytest.mark.parametrize('use_atol_array', (True, False))
@pytest.mark.parametrize('use_large_timespan', (True, False))
@pytest.mark.parametrize('use_njit_always', (False,))
@pytest.mark.parametrize('use_args', (True, False))
@pytest.mark.parametrize('use_scipy_style', (True, False))
def test_pysolve_ivp(use_scipy_style, use_args, use_njit_always,
                     use_large_timespan, use_atol_array, use_rtol_array, use_different_tols, integration_method,
                     first_step, max_step, capture_extra):
    """Check that the pysolve_ivp function is able to run with various changes to its arguments. """
    global RK23_TESTED
    global njit_rk23_tested
    global njit_rk45_tested
    global njit_DOP853_tested

    # To reduce number of tests, only test RK23 once. 
    if RK23_TESTED and SKIP_SOME_RK23_TESTS and (integration_method=="RK23"):
        pytest.skip("Skipping Some RK23 Tests (just to reduce number of tests).")
    else:
        RK23_TESTED = True

    if use_args:
        if use_scipy_style:
            if capture_extra:
                diffeq_to_use = diffeq_scipy_style_args_extra
            else:
                diffeq_to_use = diffeq_scipy_style_args
        else:
            if capture_extra:
                diffeq_to_use = diffeq_args_extra
            else:
                diffeq_to_use = diffeq_args
    else:
        if use_scipy_style:
            if capture_extra:
                diffeq_to_use = diffeq_scipy_style_extra
            else:
                diffeq_to_use = diffeq_scipy_style
        else:
            if capture_extra:
                diffeq_to_use = diffeq_extra
            else:
                diffeq_to_use = diffeq

    if use_njit_always:
        diffeq_to_use = njit(diffeq_to_use)
    else:
        if (not njit_rk23_tested) and (integration_method=="RK23"):
            diffeq_to_use = njit(diffeq_to_use)
            njit_rk23_tested = True
        elif (not njit_rk45_tested) and (integration_method=="RK45"):
            diffeq_to_use = njit(diffeq_to_use)
            njit_rk45_tested = True
        elif (not njit_DOP853_tested) and (integration_method=="DOP853"):
            diffeq_to_use = njit(diffeq_to_use)
            njit_DOP853_tested = True

    if use_atol_array:
        atols_use = atols
    else:
        atols_use = atol
    if use_rtol_array:
        rtols_use = rtols
    else:
        rtols_use = rtol
    
    if use_different_tols:
        # Check that it can run with smaller tolerances
        atols_use = atols_use / 100.0
        rtols_use = rtols_use / 100.0
    
    if use_large_timespan:
        time_span_touse = time_span_large
    else:
        time_span_touse = time_span

    if use_args:
        args_touse = args
    else:
        args_touse = None
    
    if capture_extra:
        num_extra = 2
    else:
        num_extra = 0

    result = \
        pysolve_ivp(diffeq_to_use, time_span_touse, initial_conds,
                    method=integration_method,
                    args=args_touse, rtol=rtols_use, atol=atols_use,
                    num_extra=num_extra, first_step=first_step, max_step=max_step,
                    pass_dy_as_arg=(not use_scipy_style))

    assert isinstance(result, WrapCySolverResult)
    assert result.success
    assert result.error_code == 1
    assert result.size > 1
    assert result.steps_taken > 1
    assert result.message == "Integration completed without issue."
    # Check that the ndarrays make sense
    assert type(result.t) == np.ndarray
    assert result.t.dtype == np.float64
    assert result.y.dtype == np.float64
    assert result.t.size > 1
    assert result.t.size == result.y[0].size
    assert len(result.y.shape) == 2
    assert result.y[0].size == result.y[1].size
    assert result.t.size == result.size
    assert result.y[0].size == result.size

    if capture_extra:
        assert result.y.shape[0] == 4
        assert result.y[2].size == result.y[1].size
        assert result.y[3].size == result.y[1].size
    else:
        assert result.y.shape[0] == 2

    assert type(result.message) == str



def test_pysolve_ivp_errors():
    """ Test that the correct exceptions are raised when incorrect input is provided. """

    with pytest.raises(NotImplementedError) as e_info:
        # check for unsupported integrator
        result = pysolve_ivp(diffeq_scipy_style, time_span, initial_conds,
                        method="FakeIntegrationMethod",
                        args=None, rtol=rtol, atol=atol,
                        pass_dy_as_arg=False)
    
    with pytest.raises(AttributeError) as e_info:
        # Check for bad first step
        result = pysolve_ivp(diffeq_scipy_style, time_span, initial_conds,
                        method="RK23",
                        args=None, rtol=rtol, atol=atol,
                        first_step = -10.0,
                        pass_dy_as_arg=False)
    
    with pytest.raises(AttributeError) as e_info:
        # Check for bad max step
        result = pysolve_ivp(diffeq_scipy_style, time_span, initial_conds,
                        method="RK23",
                        args=None, rtol=rtol, atol=atol,
                        max_step = -10.0,
                        pass_dy_as_arg=False)
        
    # Run integrator with a too small maximum number of steps. No exceptions should be raised but the integration should
    # not be successful
    result = pysolve_ivp(diffeq_scipy_style, time_span, initial_conds,
                        method="RK23",
                        args=None, rtol=rtol, atol=atol,
                        max_num_steps=5,
                        pass_dy_as_arg=False)

    assert not result.success
    assert result.error_code == -2
    assert result.message == "Maximum number of steps (set by user) exceeded during integration."

    # Do the same thing but now for max ram
    result = pysolve_ivp(diffeq_scipy_style, time_span, initial_conds,
                        method="RK23",
                        args=None, rtol=rtol, atol=atol,
                        max_ram_MB=0.0001,
                        pass_dy_as_arg=False)
    
    assert not result.success
    assert result.error_code == -3
    assert result.message == "Maximum number of steps (set by system architecture) exceeded during integration."

    # Do an integration with tolerances that are just way too small for the method
    result = pysolve_ivp(diffeq_stiff, time_span, initial_conds,
                        method="RK45",
                        args=None, rtol=1.0e-20, atol=1.0e-22,
                        pass_dy_as_arg=False)
    
    assert not result.success
    assert result.error_code == -1
    assert result.message == "Error in step size calculation:\n\tRequired step size is less than spacing between numbers."

@pytest.mark.parametrize('integration_method', ("RK23", "RK45", "DOP853"))
@pytest.mark.parametrize('t_eval_end', (None, 0.5, 1.0))
@pytest.mark.parametrize('test_dense_output', (False, True))
@pytest.mark.parametrize('backward_integrate', (False, True))
def test_pysolve_ivp_accuracy(integration_method, t_eval_end, test_dense_output, backward_integrate):
    #Check that the cython function solver is able to reproduce a known functions integral with reasonable accuracy

    # TODO: This is only checking one equation. Add other types of diffeqs to provide better coverage.

    # Differential Equation
    @njit
    def diffeq_accuracy(dy, t, y):
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

    # Check if we should test t_eval
    if t_eval_end is None:
        t_eval = None
    else:
        t_eval = np.linspace(0.0, t_eval_end, 50)
    
    if backward_integrate:
        time_span_ = (time_span_[1], time_span_[0])
        if t_eval is not None:
            t_eval = np.ascontiguousarray(t_eval[::-1])

    result = \
        pysolve_ivp(diffeq_accuracy, time_span_, y0, method=integration_method, t_eval=t_eval, dense_output=test_dense_output,
                    rtol=1.0e-8, atol=1.0e-9, pass_dy_as_arg=True)
    
    # Use the integrator's time domain to build a correct solution
    real_answer = correct_answer(result.t, c1, c2)

    if t_eval is not None:
        assert result.t.size == t_eval.size
        assert np.allclose(result.t, t_eval)

    if integration_method == "RK23":
        check_rtol = 1.0e-3
        check_atol = 1.0e-6
    elif integration_method == "DOP853":
        check_rtol = 1.0e-5
        check_atol = 1.0e-8
    else:
        check_rtol = 1.0e-4
        check_atol = 1.0e-7
    
    if backward_integrate:
        pytest.skip("Backward integrating is not working well.")

    assert np.allclose(result.y, real_answer, rtol=check_rtol, atol=check_atol)

    if test_dense_output:
        # Check that dense output is working and that it gives decent results
        # Check with a float
        y_array = result(0.5)
        assert type(y_array) == np.ndarray
        assert y_array.shape == (2, 1)

        # Check with array
        t_array = np.linspace(0.1, 0.4, 10)
        y_array = result(t_array)
        assert type(y_array) == np.ndarray
        assert y_array.shape == (2, 10)

        # Check accuracy
        y_array_real = correct_answer(t_array, c1, c2)
        assert np.allclose(y_array, y_array_real, rtol=check_rtol, atol=check_atol)

    # Check the accuracy of the results
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(result.t, result.y[0], 'r', label='PySolver')
    # ax.plot(result.t, result.y[1], 'r:')
    # ax.plot(result.t, real_answer[0], 'b', label='Analytic')
    # ax.plot(result.t, real_answer[1], 'b:')
    # plt.show()

@pytest.mark.parametrize('integration_method', ("RK23", "RK45", "DOP853"))
def test_pysolve_ivp_readonly(integration_method):
    #Check that the cython function solver is able to reproduce a known functions integral with reasonable accuracy

    @njit
    def diffeq_accuracy(dy, t, y):
        dy[0] = np.sin(t) - y[1]  # dydt = sin(t) - x(t)
        dy[1] = np.cos(t) + y[0]  # dxdt = cos(t) + y(t)

    # Initial Conditions
    # y=0 --> c2 = + 1/2
    c2 = 0.5
    # x=1 --> c1 = + 1
    c1 = 1.0
    y0 = np.asarray((0., 1.), dtype=np.float64)
    # Make readonly
    y0.setflags(write=False)
    time_span_ = (0., 10.)

    result = \
        pysolve_ivp(diffeq_accuracy, time_span_, y0, method=integration_method,
                    rtol=1.0e-8, atol=1.0e-9, pass_dy_as_arg=True)
    
    assert isinstance(result, WrapCySolverResult)
    assert result.success
    assert result.error_code == 1
    assert result.size > 1
    assert result.message == "Integration completed without issue."
    # Check that the ndarrays make sense
    assert type(result.t) == np.ndarray
    assert result.t.dtype == np.float64
    assert result.y.dtype == np.float64
    assert result.t.size > 1
    assert result.t.size == result.y[0].size
    assert len(result.y.shape) == 2
    assert result.y[0].size == result.y[1].size
    assert result.t.size == result.size
    assert result.y[0].size == result.size
    assert result.y.shape[0] == 2
    assert type(result.message) == str
