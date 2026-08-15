"""Tests that a reused solution stays consistent with the problem it is asked to solve.

A `CySolverResult` derives several properties (`num_y`, `num_dy`, `capture_extra`, ...) from the
configuration it is given. Those derived properties have to be rebuilt whenever the solution is
reused, otherwise the second run silently keeps the first run's problem shape.
"""

import numpy as np
import pytest
from numba import njit

from CyRK import pysolve_ivp
from CyRK.cy.cysolver_test import cytester

time_span = (0., 1.)
rtol = 1.0e-10
atol = 1.0e-12


@njit
def decay_2(dy, t, y):
    dy[0] = -1.0 * y[0]
    dy[1] = -2.0 * y[1]


@njit
def decay_3(dy, t, y):
    dy[0] = -1.0 * y[0]
    dy[1] = -2.0 * y[1]
    dy[2] = -3.0 * y[2]


@njit
def decay_2_with_extra(dy, t, y):
    dy[0] = -1.0 * y[0]
    dy[1] = -2.0 * y[1]
    # Constant extra outputs so that they are trivial to check.
    dy[2] = 111.0
    dy[3] = 222.0


def decay_answer(decay_rates, t):
    """Exact solution of the decay problems for y(0) = 1."""
    return np.exp(-np.asarray(decay_rates, dtype=np.float64) * t)


@pytest.mark.parametrize('integration_method', ("RK23", "RK45", "DOP853"))
def test_pysolve_reuse_rejects_changed_num_y(integration_method):
    """Reusing a solution for a problem with a different number of dependent variables must fail.

    The solver's dependent variable storage is shared with numpy arrays that were built for the
    previous problem's size, so the user has to build a new solver instead.
    """
    result = pysolve_ivp(decay_2, time_span, np.asarray((1., 1.), dtype=np.float64, order='C'),
                         method=integration_method, rtol=rtol, atol=atol, pass_dy_as_arg=True)
    assert result.success

    # Growing the problem.
    with pytest.raises(AttributeError, match="different number of dependent variables"):
        pysolve_ivp(decay_3, time_span, np.asarray((1., 1., 1.), dtype=np.float64, order='C'),
                    method=integration_method, rtol=rtol, atol=atol, pass_dy_as_arg=True,
                    solution_reuse=result)

    # Shrinking the problem.
    with pytest.raises(AttributeError, match="different number of dependent variables"):
        pysolve_ivp(decay_2, time_span, np.asarray((1.,), dtype=np.float64, order='C'),
                    method=integration_method, rtol=rtol, atol=atol, pass_dy_as_arg=True,
                    solution_reuse=result)

    # The rejected calls must not have disturbed the solution that was passed in.
    assert result.success
    assert result.num_y == 2
    assert np.allclose(result.y[:, -1], decay_answer((1., 2.), time_span[1]))


def test_pysolve_reuse_after_rejection_still_works():
    """A rejected reuse must leave the solution usable for a problem of the original size."""
    result = pysolve_ivp(decay_2, time_span, np.asarray((1., 1.), dtype=np.float64, order='C'),
                         rtol=rtol, atol=atol, pass_dy_as_arg=True)

    with pytest.raises(AttributeError):
        pysolve_ivp(decay_3, time_span, np.asarray((1., 1., 1.), dtype=np.float64, order='C'),
                    rtol=rtol, atol=atol, pass_dy_as_arg=True, solution_reuse=result)

    result = pysolve_ivp(decay_2, (0., 2.), np.asarray((1., 1.), dtype=np.float64, order='C'),
                         rtol=rtol, atol=atol, pass_dy_as_arg=True, solution_reuse=result)

    assert result.success
    assert result.num_y == 2
    assert np.allclose(result.y[:, -1], decay_answer((1., 2.), 2.0))


def test_pysolve_reuse_changed_num_y_allowed_with_new_solver():
    """Building a new solver is the supported way to change the number of dependent variables."""
    result = pysolve_ivp(decay_2, time_span, np.asarray((1., 1.), dtype=np.float64, order='C'),
                         rtol=rtol, atol=atol, pass_dy_as_arg=True)
    assert result.success

    result = pysolve_ivp(decay_3, time_span, np.asarray((1., 1., 1.), dtype=np.float64, order='C'),
                         rtol=rtol, atol=atol, pass_dy_as_arg=True)

    assert result.success
    assert result.num_y == 3
    assert np.allclose(result.y[:, -1], decay_answer((1., 2., 3.), time_span[1]))


@pytest.mark.parametrize('integration_method', ("RK23", "RK45", "DOP853"))
def test_pysolve_reuse_changed_num_extra(integration_method):
    """Changing the number of extra outputs on a reused solution must be picked up."""
    y0 = np.asarray((1., 1.), dtype=np.float64, order='C')

    result = pysolve_ivp(decay_2, time_span, y0, method=integration_method,
                         rtol=rtol, atol=atol, pass_dy_as_arg=True)
    assert result.success
    assert result.num_dy == 2
    assert result.y.shape[0] == 2

    # Turn extra output on.
    result = pysolve_ivp(decay_2_with_extra, time_span, y0, method=integration_method,
                         rtol=rtol, atol=atol, num_extra=2, pass_dy_as_arg=True,
                         solution_reuse=result)
    assert result.success
    assert result.num_dy == 4
    assert result.y.shape[0] == 4
    assert np.allclose(result.y[:2, -1], decay_answer((1., 2.), time_span[1]))
    assert np.allclose(result.y[2], 111.0)
    assert np.allclose(result.y[3], 222.0)

    # And back off again.
    result = pysolve_ivp(decay_2, time_span, y0, method=integration_method,
                         rtol=rtol, atol=atol, pass_dy_as_arg=True, solution_reuse=result)
    assert result.success
    assert result.num_dy == 2
    assert result.y.shape[0] == 2
    assert np.allclose(result.y[:, -1], decay_answer((1., 2.), time_span[1]))


def test_pysolve_reuse_changed_t_eval():
    """Turning `t_eval` on and off across reuses must be picked up."""
    y0 = np.asarray((1., 1.), dtype=np.float64, order='C')
    t_eval = np.linspace(0.0, 1.0, 17)

    result = pysolve_ivp(decay_2, time_span, y0, rtol=rtol, atol=atol, pass_dy_as_arg=True)
    assert result.success
    steps_without_t_eval = result.size

    result = pysolve_ivp(decay_2, time_span, y0, rtol=rtol, atol=atol, t_eval=t_eval,
                         pass_dy_as_arg=True, solution_reuse=result)
    assert result.success
    assert result.size == t_eval.size
    assert np.allclose(result.t, t_eval)

    result = pysolve_ivp(decay_2, time_span, y0, rtol=rtol, atol=atol, pass_dy_as_arg=True,
                         solution_reuse=result)
    assert result.success
    assert result.size == steps_without_t_eval


def test_cysolve_reuse_changed_num_y():
    """`cysolve_ivp` builds its configuration through `update_properties`, so it can be reused
    across problems of different sizes. Check that this keeps working."""
    # Test diffeq 0 has two dependent variables; diffeq 3 (Lorenz) has three.
    result = cytester(0, time_span, np.asarray((20., 20.), dtype=np.float64, order='C'),
                      method='rk45', rtol=rtol, atol=atol)
    assert result.success
    assert result.num_y == 2

    y0_lorenz = np.asarray((1., 0., 0.), dtype=np.float64, order='C')
    reused_result = cytester(3, time_span, y0_lorenz, method='rk45', rtol=rtol, atol=atol,
                             solution_reuse=result)
    fresh_result = cytester(3, time_span, y0_lorenz, method='rk45', rtol=rtol, atol=atol)

    assert reused_result.success
    assert reused_result.num_y == 3
    assert np.allclose(reused_result.y[:, -1], fresh_result.y[:, -1])
