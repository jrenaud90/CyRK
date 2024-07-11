import numpy as np
import pytest

from CyRK.cy.cysolverNew import WrapCySolverResult
from CyRK.cy.cysolverNew_test import cytester

args = (0.01, 0.02)

initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8

rtols = np.asarray((1.0e-7, 1.0e-8), dtype=np.float64, order='C')
atols = np.asarray((1.0e-8, 1.0e-9), dtype=np.float64, order='C')


@pytest.mark.filterwarnings("error")  # Some exceptions get propagated via cython as warnings; we want to make sure the lead to crashes.
@pytest.mark.parametrize('capture_extra', (True, False))
@pytest.mark.parametrize('max_step', (1.0, 100_000.0))
@pytest.mark.parametrize('first_step', (0.0, 0.01))
@pytest.mark.parametrize('integration_method', (0, 1, 2))
@pytest.mark.parametrize('use_different_tols', (True, False))
@pytest.mark.parametrize('use_rtol_array', (True, False))
@pytest.mark.parametrize('use_atol_array', (True, False))
@pytest.mark.parametrize('use_large_timespan', (True, False))
@pytest.mark.parametrize('use_args', (True, False))
def test_pysolve_ivp(use_args,
                     use_large_timespan, use_atol_array, use_rtol_array, use_different_tols, integration_method,
                     first_step, max_step, capture_extra):
    """Check that the pysolve_ivp function is able to run with various changes to its arguments. """

    if use_atol_array:
        atols_float = np.nan
        atols_array = atols
    else:
        atols_float = atol
        atols_array = None
    if use_rtol_array:
        rtols_float = np.nan
        rtols_array = rtols
    else:
        rtols_float = rtol
        rtols_array = None
    
    if use_different_tols:
        # Check that it can run with smaller tolerances
        if rtols_float is not None:
            rtols_float /= 100.0
        if rtols_array is not None:
            rtols_array /= 100.0
        if atols_float is not None:
            atols_float /= 100.0
        if atols_array is not None:
            atols_array /= 100.0
    
    if use_large_timespan:
        time_span_touse = time_span_large
    else:
        time_span_touse = time_span

    if use_args:
        args_touse = np.asarray(args, dtype=np.float64, order='C')
    else:
        args_touse = None
    
    if capture_extra:
        diffeq_num = 2
    else:
        diffeq_num = 0

    result = \
        cytester(diffeq_num, time_span_touse, initial_conds, args=args_touse,
                 method=integration_method,
                 rtol=rtols_float, atol=atols_float, rtol_array=rtols_array, atol_array=atols_array,
                 max_step=max_step, first_step=first_step)

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

    if capture_extra:
        assert result.y.shape[0] == 4
        assert result.y[2].size == result.y[1].size
        assert result.y[3].size == result.y[1].size
    else:
        assert result.y.shape[0] == 2

    assert type(result.message) == str


@pytest.mark.parametrize('integration_method', (0, 1, 2))
def test_pysolve_ivp_accuracy(integration_method):
    #Check that the cython function solver is able to reproduce a known functions integral with reasonable accuracy

    # TODO: This is only checking one equation. Add other types of diffeqs to provide better coverage.

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
    y0 = np.asarray((0., 1.), dtype=np.float64, order='C')
    time_span_ = (0., 10.)

    result = \
        cytester(1, time_span_, y0,
                 method=integration_method,
                 rtol=1.0e-8, atol=1.0e-9)
    
    # Use the integrator's time domain to build a correct solution
    real_answer = correct_answer(result.t, c1, c2)

    if integration_method == 0:
        assert np.allclose(result.y, real_answer, rtol=1.0e-3, atol=1.0e-6)
    elif integration_method == 1:
        assert np.allclose(result.y, real_answer, rtol=1.0e-4, atol=1.0e-7)
    else:
        assert np.allclose(result.y, real_answer, rtol=1.0e-5, atol=1.0e-8)

    # Check the accuracy of the results
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(result.t, result.y[0], 'r', label='PySolver')
    # ax.plot(result.t, result.y[1], 'r:')
    # ax.plot(result.t, real_answer[0], 'b', label='Analytic')
    # ax.plot(result.t, real_answer[1], 'b:')
    # plt.show()
