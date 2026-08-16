"""Tests for CyRK's implicit integrators, BDF and LSODA.

The problems used here are stiff on purpose: they are cheap for an implicit method but force an
explicit method to take very small steps, which is what makes the comparisons meaningful.
"""

import numpy as np
import pytest
from numba import njit

from CyRK import pysolve_ivp, ODEMethod
from CyRK.cy.cysolver_test import cytester
from CyRK.cy.pyhelpers import find_ode_method_int

IMPLICIT_METHODS = ("BDF", "LSODA")

# Stiffness of the test problem below. Large enough that RK45 struggles, small enough to stay fast.
STIFF_ALPHA = 1.0e4


@njit
def stiff_diffeq(dy, t, y):
    """Stiff problem whose first component is driven onto cos(t) with a fast transient."""
    dy[0] = -STIFF_ALPHA * (y[0] - np.cos(t)) - np.sin(t)
    dy[1] = y[0] - y[1]


def stiff_answer(t):
    """Exact solution of `stiff_diffeq` for y(0) = (1, 0)."""
    y = np.empty((2, np.asarray(t).size), dtype=np.float64)
    y[0] = np.cos(t)
    y[1] = 0.5 * (np.cos(t) + np.sin(t)) - 0.5 * np.exp(-t)
    return y


STIFF_Y0 = np.asarray((1.0, 0.0), dtype=np.float64, order='C')
STIFF_TIME_SPAN = (0.0, 10.0)


@njit
def oscillator_diffeq(dy, t, y):
    """Non-stiff oscillator used where a problem has to be stable in both time directions."""
    dy[0] = np.sin(t) - y[1]
    dy[1] = np.cos(t) + y[0]


def oscillator_answer(t):
    """Exact solution of `oscillator_diffeq` for y(0) = (0, 1)."""
    t = np.asarray(t, dtype=np.float64)
    y = np.empty((2, t.size), dtype=np.float64)
    y[0] = -np.sin(t)
    y[1] = np.sin(t) + np.cos(t)
    return y


@njit
def diffusion_diffeq(dy, t, y):
    """Discretized diffusion on a line; the Jacobian is tridiagonal."""
    num_y = y.size
    for i in range(num_y):
        left = 0.0 if i == 0 else y[i - 1]
        right = 0.0 if i == (num_y - 1) else y[i + 1]
        dy[i] = 50.0 * (left - 2.0 * y[i] + right)


@njit
def stiff_extra_diffeq(dy, t, y):
    """Same as `stiff_diffeq` but also reports two intermediate values as extra output."""
    extra_0 = np.cos(t)
    extra_1 = y[0] - y[1]
    dy[0] = -STIFF_ALPHA * (y[0] - extra_0) - np.sin(t)
    dy[1] = extra_1
    dy[2] = extra_0
    dy[3] = extra_1


def test_implicit_methods_are_registered():
    """The new methods must be reachable by name and hold stable enum values."""
    assert int(ODEMethod.BDF) == 6
    assert int(ODEMethod.LSODA) == 7
    assert find_ode_method_int('bdf') == int(ODEMethod.BDF)
    assert find_ode_method_int('LSODA') == int(ODEMethod.LSODA)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_accuracy(integration_method):
    """Both implicit methods should reproduce the exact solution of a stiff problem."""
    result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                         rtol=1.0e-9, atol=1.0e-11, pass_dy_as_arg=True)

    assert result.success
    assert result.message == "Integration completed without issue."
    assert result.size > 1
    assert np.allclose(result.y, stiff_answer(result.t), rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_beats_explicit_on_stiff_problem(integration_method):
    """The whole point of an implicit method: far fewer steps when the problem is stiff."""
    implicit_result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                                  rtol=1.0e-8, atol=1.0e-10, pass_dy_as_arg=True)
    explicit_result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method='RK45',
                                  rtol=1.0e-8, atol=1.0e-10, pass_dy_as_arg=True)

    assert implicit_result.success
    assert explicit_result.success
    assert implicit_result.steps_taken < (explicit_result.steps_taken / 10)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_dense_output(integration_method):
    """The interpolants for the multi-step methods should match the exact solution."""
    result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                         rtol=1.0e-9, atol=1.0e-11, dense_output=True, pass_dy_as_arg=True)
    assert result.success

    # Single time value.
    y_single = result(4.25)
    assert y_single.shape == (2, 1)
    assert np.allclose(y_single[:, 0], stiff_answer(4.25)[:, 0], rtol=1.0e-5, atol=1.0e-8)

    # Vectorized call. The first fast transient is skipped since no interpolant can resolve a
    # boundary layer that the solver stepped over.
    t_array = np.linspace(0.01, 9.9, 100)
    y_array = result(t_array)
    assert y_array.shape == (2, 100)
    assert np.allclose(y_array, stiff_answer(t_array), rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_t_eval(integration_method):
    """Requested output times should be hit exactly and interpolated accurately."""
    t_eval = np.linspace(0.01, 9.9, 75)
    result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                         rtol=1.0e-9, atol=1.0e-11, t_eval=t_eval, pass_dy_as_arg=True)

    assert result.success
    assert result.t.size == t_eval.size
    assert np.allclose(result.t, t_eval)
    assert np.allclose(result.y, stiff_answer(result.t), rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_backward_integration(integration_method):
    """Integrating from the end of the domain back to the start should recover the start state.

    A non-stiff problem is used here on purpose: running a stiff problem in reverse turns its fast
    decaying mode into a fast growing one, which no solver can follow.
    """
    y_end = np.asarray(oscillator_answer(10.0)[:, 0], dtype=np.float64, order='C')
    result = pysolve_ivp(oscillator_diffeq, (10.0, 0.0), y_end, method=integration_method,
                         rtol=1.0e-10, atol=1.0e-12, pass_dy_as_arg=True)

    assert result.success
    assert result.t[-1] == 0.0
    assert np.allclose(result.y[:, -1], oscillator_answer(0.0)[:, 0], rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
@pytest.mark.parametrize('use_arrays', (True, False))
def test_implicit_tolerance_arrays(integration_method, use_arrays):
    """Per-variable tolerances have to work the same way that they do for the RK methods."""
    if use_arrays:
        rtol = np.asarray((1.0e-9, 1.0e-10), dtype=np.float64, order='C')
        atol = np.asarray((1.0e-11, 1.0e-12), dtype=np.float64, order='C')
    else:
        rtol = 1.0e-9
        atol = 1.0e-11

    result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                         rtol=rtol, atol=atol, pass_dy_as_arg=True)

    assert result.success
    assert np.allclose(result.y, stiff_answer(result.t), rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_extra_output(integration_method):
    """Extra output must be recorded at the accepted state of each step."""
    y0 = np.asarray((1.0, 0.0), dtype=np.float64, order='C')
    result = pysolve_ivp(stiff_extra_diffeq, STIFF_TIME_SPAN, y0, method=integration_method,
                         rtol=1.0e-9, atol=1.0e-11, num_extra=2, pass_dy_as_arg=True)

    assert result.success
    assert result.y.shape[0] == 4
    # The extra outputs are cos(t) and y0 - y1, both known exactly.
    exact = stiff_answer(result.t)
    assert np.allclose(result.y[2], np.cos(result.t), rtol=1.0e-5, atol=1.0e-8)
    assert np.allclose(result.y[3], exact[0] - exact[1], rtol=1.0e-5, atol=1.0e-7)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
@pytest.mark.parametrize('terminate', (True, False))
def test_implicit_events(integration_method, terminate):
    """Event detection runs off the interpolants, so it has to work for the implicit methods too."""

    def event_zero_crossing(t, y):
        # Triggers whenever the first component crosses zero, which cos(t) does at pi/2 and 3pi/2.
        return y[0]

    if terminate:
        event_zero_crossing.terminal = 1

    result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                         rtol=1.0e-9, atol=1.0e-11, events=event_zero_crossing, pass_dy_as_arg=True)

    assert result.success
    assert result.num_events == 1
    assert result.t_events[0].size > 0
    # The first crossing of cos(t) is at pi / 2.
    assert np.isclose(result.t_events[0][0], np.pi / 2.0, rtol=1.0e-6, atol=1.0e-8)

    if terminate:
        assert result.event_terminated
        assert result.event_terminate_index == 0
        assert result.t_events[0].size == 1
        assert np.isclose(result.t[-1], np.pi / 2.0, rtol=1.0e-6, atol=1.0e-8)
    else:
        assert not result.event_terminated
        assert result.t_events[0].size > 1


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_cysolve_ivp(integration_method):
    """The C-level solver entry point must accept the implicit methods as well."""
    result = cytester(1, (0.0, 10.0), np.asarray((0.0, 1.0), dtype=np.float64, order='C'),
                      method=integration_method.lower(), rtol=1.0e-9, atol=1.0e-11)

    assert result.success
    assert result.size > 1
    # y0 = -sin(t) + cos(t) / 2 - cos(t) / 2 ... compare against a tight RK45 run instead.
    reference = cytester(1, (0.0, 10.0), np.asarray((0.0, 1.0), dtype=np.float64, order='C'),
                         method='rk45', rtol=1.0e-12, atol=1.0e-13)
    assert np.allclose(result.y[:, -1], reference.y[:, -1], rtol=1.0e-6, atol=1.0e-8)


def test_lsoda_banded_jacobian():
    """A banded Jacobian should give the same answer as the dense one, with less work per step."""
    num_y = 30
    y0 = np.zeros(num_y, dtype=np.float64, order='C')
    y0[num_y // 2] = 100.0

    dense_result = pysolve_ivp(diffusion_diffeq, (0.0, 1.0), y0, method='LSODA',
                               rtol=1.0e-9, atol=1.0e-11, pass_dy_as_arg=True)
    banded_result = pysolve_ivp(diffusion_diffeq, (0.0, 1.0), y0, method='LSODA',
                                rtol=1.0e-9, atol=1.0e-11, lband=1, uband=1, pass_dy_as_arg=True)

    assert dense_result.success
    assert banded_result.success
    assert np.allclose(banded_result.y[:, -1], dense_result.y[:, -1], rtol=1.0e-6, atol=1.0e-8)


def test_lsoda_min_step():
    """A minimum step size should keep LSODA from refining the step below it."""
    unbounded_result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method='LSODA',
                                   rtol=1.0e-12, atol=1.0e-14, pass_dy_as_arg=True)
    bounded_result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method='LSODA',
                                 rtol=1.0e-12, atol=1.0e-14, min_step=0.05, pass_dy_as_arg=True)

    assert unbounded_result.success
    # With a floor under the step size the solver cannot resolve the initial transient, so it
    # needs far fewer steps and gives up accuracy in return.
    assert bounded_result.steps_taken < unbounded_result.steps_taken


@pytest.mark.parametrize('integration_method', ("RK23", "RK45", "DOP853", "BDF"))
def test_lsoda_only_options_are_rejected_elsewhere(integration_method):
    """`min_step`, `lband`, and `uband` are LSODA-only and must not be silently ignored."""
    with pytest.raises(AttributeError):
        pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                    min_step=1.0e-6, pass_dy_as_arg=True)
    with pytest.raises(AttributeError):
        pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                    lband=1, pass_dy_as_arg=True)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_solution_reuse(integration_method):
    """Re-running through the same solution object must reset the solver state cleanly."""
    result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                         rtol=1.0e-9, atol=1.0e-11, pass_dy_as_arg=True)
    assert result.success
    first_size = result.size

    for _ in range(3):
        result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                             rtol=1.0e-9, atol=1.0e-11, pass_dy_as_arg=True,
                             solution_reuse=result)
        assert result.success
        assert result.size == first_size
        assert np.allclose(result.y, stiff_answer(result.t), rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_first_step_and_max_step(integration_method):
    """User-provided step size limits should be honored by the implicit methods."""
    result = pysolve_ivp(stiff_diffeq, STIFF_TIME_SPAN, STIFF_Y0, method=integration_method,
                         rtol=1.0e-8, atol=1.0e-10, first_step=1.0e-6, max_step=0.1,
                         pass_dy_as_arg=True)

    assert result.success
    assert np.allclose(result.y, stiff_answer(result.t), rtol=1.0e-4, atol=1.0e-7)
    # With the step size capped there must be at least (t_end - t_start) / max_step steps.
    assert result.steps_taken >= 100
