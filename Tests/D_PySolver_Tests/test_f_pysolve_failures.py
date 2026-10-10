"""Tests that a Python diffeq or event that fails stops the integration instead of being silently ignored.

The C++ solvers call the Python functions through a `noexcept` hook, so an exception can not travel through them.
`PySolver` stores it, stops the solver, and raises it again once the solver returns. A diffeq that returns NaN or
inf must end the integration unsuccessfully, for every method (LSODA once accepted NaN steps and reported success).
"""

import numpy as np
import pytest

from CyRK import pysolve_ivp, CyrkErrorCodes
from CyRK.cy.pysolver import PySolver

METHODS = ("RK23", "RK45", "DOP853", "Tsit5", "Vern7", "Vern8", "BDF", "LSODA", "Radau")
time_span = (0.0, 100.0)
switch_time = 10.0
y0 = np.asarray((1.0,), dtype=np.float64, order='C')


class DiffeqFailure(Exception):
    pass


def decay(t, y):
    return np.asarray((-y[0],), dtype=np.float64)


def raises_late(t, y):
    if t > switch_time:
        raise DiffeqFailure("raised by the diffeq")
    return decay(t, y)


def raises_at_start(t, y):
    raise DiffeqFailure("raised at the initial state")


def event_raises_late(t, y):
    if t > switch_time:
        raise DiffeqFailure("raised by the event")
    return y[0] - 1.0e-20


event_raises_late.terminal = True


@pytest.mark.parametrize('integration_method', METHODS)
def test_diffeq_exception_is_raised(integration_method):
    """An exception raised by the diffeq after the start reaches the caller."""
    with pytest.raises(DiffeqFailure, match="raised by the diffeq"):
        pysolve_ivp(raises_late, time_span, y0, method=integration_method, rtol=1.0e-6, atol=1.0e-9)


@pytest.mark.parametrize('integration_method', METHODS)
def test_event_exception_is_raised(integration_method):
    """An exception raised by an event function reaches the caller instead of counting as the event firing."""
    with pytest.raises(DiffeqFailure, match="raised by the event"):
        pysolve_ivp(decay, time_span, y0, method=integration_method, rtol=1.0e-6, atol=1.0e-9,
                    events=[event_raises_late])


def test_exception_at_the_initial_state_is_raised():
    with pytest.raises(DiffeqFailure, match="initial state"):
        pysolve_ivp(raises_at_start, time_span, y0)


def test_exception_with_dy_argument_and_args():
    def dy_with_args(dy, t, y, rate):
        if t > switch_time:
            raise DiffeqFailure("raised with pass_dy_as_arg")
        dy[0] = -rate * y[0]

    with pytest.raises(DiffeqFailure, match="pass_dy_as_arg"):
        pysolve_ivp(dy_with_args, time_span, y0, method="BDF", args=(1.0,), pass_dy_as_arg=True)


def test_keyboard_interrupt_stops_the_integration():
    def interrupted(t, y):
        if t > switch_time:
            raise KeyboardInterrupt
        return decay(t, y)

    with pytest.raises(KeyboardInterrupt):
        pysolve_ivp(interrupted, time_span, y0, method="LSODA")


def test_diffeq_returning_too_few_values_raises():
    """With extra outputs the diffeq must return num_y + num_extra values; fewer would be read past their end."""
    with pytest.raises(ValueError, match="2 are expected"):
        pysolve_ivp(decay, time_span, y0, num_extra=1)


def test_exception_during_a_dense_call_is_raised():
    """With extra outputs every dense call evaluates the diffeq again, so its exception must reach that call."""
    state = {"fail": False}

    def with_extra(t, y):
        if state["fail"]:
            raise DiffeqFailure("raised during a dense call")
        return np.asarray((-y[0], 2.0 * y[0]), dtype=np.float64)

    solution = pysolve_ivp(with_extra, (0.0, 1.0), y0, num_extra=1, dense_output=True)
    assert solution.success
    state["fail"] = True
    with pytest.raises(DiffeqFailure):
        solution.call(0.5)
    with pytest.raises(DiffeqFailure):
        solution.call_vectorize(np.asarray((0.2, 0.4), dtype=np.float64))


@pytest.mark.parametrize('integration_method', METHODS)
def test_solution_arrays_after_an_exception(integration_method):
    """A solver whose diffeq raised keeps the steps it stored: at most the initial state when it raised there (the
    arrays are readable, not unset), the steps before the failure when it raised later."""
    solver = PySolver()
    with pytest.raises(DiffeqFailure):
        pysolve_ivp(raises_at_start, time_span, y0, method=integration_method, solution_reuse=solver)
    t, y = np.asarray(solver.t), np.asarray(solver.y)
    assert t.size == solver.size <= 1 and y.shape == (1, t.size)
    if t.size == 1:
        assert t[0] == time_span[0] and y[0, 0] == y0[0]

    solver = PySolver()
    with pytest.raises(DiffeqFailure):
        pysolve_ivp(raises_late, time_span, y0, method=integration_method, solution_reuse=solver)
    t, y = np.asarray(solver.t), np.asarray(solver.y)
    assert t.size > 1 and y.shape == (1, t.size)
    assert t[0] == time_span[0] and t[-1] <= switch_time + 10.0


def test_reused_solver_after_an_exception():
    """A solver whose run raised solves the next problem normally."""
    solver = PySolver()
    with pytest.raises(DiffeqFailure):
        pysolve_ivp(raises_at_start, time_span, y0, solution_reuse=solver)
    result = pysolve_ivp(decay, (0.0, 1.0), y0, solution_reuse=solver, rtol=1.0e-10, atol=1.0e-12)
    assert result.success
    assert np.asarray(result.y)[0, -1] == pytest.approx(np.exp(-1.0), rel=1.0e-6)


def never_fires(t, y):
    return y[0] + 10.0


@pytest.mark.parametrize('integration_method', METHODS)
@pytest.mark.parametrize('bad_value', (np.nan, np.inf))
@pytest.mark.parametrize('use_event', (False, True))
def test_non_finite_rates_fail(integration_method, bad_value, use_event):
    """Rates that turn NaN or inf at t = 10 end the integration unsuccessfully near t = 10, with finite states and the
    reason for the failure.

    LSODA's weighted norms dropped NaN (a NaN step passed its error test) and its step then shrank to the spacing
    between numbers without failing. With an event, checking it after the failed step replaced the step's error status
    with `NO_ERROR` ("No errors were encountered"), and the Runge-Kutta methods stepped on until the step limit.
    """
    def turns_bad(t, y):
        if t > switch_time:
            return np.asarray((bad_value,), dtype=np.float64)
        return decay(t, y)

    events = [never_fires] if use_event else None
    result = pysolve_ivp(turns_bad, time_span, y0, method=integration_method, rtol=1.0e-6, atol=1.0e-9,
                         events=events)
    assert not result.success
    assert result.status == CyrkErrorCodes.STEP_SIZE_ERROR_SPACING
    assert np.asarray(result.t)[-1] <= switch_time + 1.0e-6
    # A failing solver must stop promptly rather than creep to the step limit.
    assert result.steps_taken < 1000
    if integration_method == "LSODA":
        # Every step LSODA saved was taken with finite rates.
        assert np.all(np.isfinite(np.asarray(result.y)))
