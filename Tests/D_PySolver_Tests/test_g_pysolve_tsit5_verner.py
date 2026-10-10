"""Method-specific checks of the Tsit5, Vern7, and Vern8 explicit Runge-Kutta methods.

SciPy has no equivalent of these methods, so their results are checked against analytic solutions and a tight DOP853
integration: the convergence order of each method and of its interpolant, the accuracy of dense output, `t_eval`, and
events, and the steps and diffeq calls of a reference implementation of the same step control.
"""

import numpy as np
import pytest
from numba import njit

from CyRK import pysolve_ivp
from CyRK.cy.cysolver_api import ODEMethod
from CyRK.cy.pyhelpers import find_ode_method_int

NEW_METHODS = ("Tsit5", "Vern7", "Vern8")
# Order of the propagated solution and of the interpolant (dense output) of each method.
METHOD_ORDER = {"Tsit5": 5, "Vern7": 7, "Vern8": 8}
INTERPOLANT_ORDER = {"Tsit5": 4, "Vern7": 7, "Vern8": 8}
# A tolerance large enough that every step is accepted, so that first_step = max_step fixes the step size.
FIXED_STEP_TOL = 1.0e3


@njit
def riccati_diffeq(dy, t, y):
    # y0 = 1 / (3 - sin(t)) solves the non-linear y0' = y0^2 cos(t); y1 = exp(1 - cos(t)) solves y1' = y1 sin(t).
    dy[0] = y[0] * y[0] * np.cos(t)
    dy[1] = y[1] * np.sin(t)


def riccati_answer(t):
    return np.array([1.0 / (3.0 - np.sin(t)), np.exp(1.0 - np.cos(t))])


KEPLER_ECCENTRICITY = 0.6


@njit
def kepler_diffeq(dy, t, y):
    radius_cubed = (y[0] * y[0] + y[1] * y[1]) ** 1.5
    dy[0] = y[2]
    dy[1] = y[3]
    dy[2] = -y[0] / radius_cubed
    dy[3] = -y[1] / radius_cubed


def kepler_answer(t):
    """Kepler orbit (unit semi-major axis and gravitational parameter) started at pericenter at t = 0."""
    mean_anomaly = np.atleast_1d(np.asarray(t, dtype=np.float64))
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(50):
        eccentric_anomaly -= ((eccentric_anomaly - KEPLER_ECCENTRICITY * np.sin(eccentric_anomaly) - mean_anomaly)
                              / (1.0 - KEPLER_ECCENTRICITY * np.cos(eccentric_anomaly)))
    anomaly_rate = 1.0 / (1.0 - KEPLER_ECCENTRICITY * np.cos(eccentric_anomaly))
    minor_axis = np.sqrt(1.0 - KEPLER_ECCENTRICITY ** 2)
    return np.array([
        np.cos(eccentric_anomaly) - KEPLER_ECCENTRICITY,
        minor_axis * np.sin(eccentric_anomaly),
        -np.sin(eccentric_anomaly) * anomaly_rate,
        minor_axis * np.cos(eccentric_anomaly) * anomaly_rate])


@njit
def oscillator_diffeq(dy, t, y):
    dy[0] = y[1]
    dy[1] = -y[0]


def test_method_registration():
    """The new methods append to `ODEMethod` (whose integer values are public API) and are found by name."""
    assert int(ODEMethod.TSIT5) == 9
    assert int(ODEMethod.VERN7) == 10
    assert int(ODEMethod.VERN8) == 11
    for name, method in zip(NEW_METHODS, (ODEMethod.TSIT5, ODEMethod.VERN7, ODEMethod.VERN8)):
        assert find_ode_method_int(name) == int(method)
        assert find_ode_method_int(name.upper()) == int(method)
        result = pysolve_ivp(oscillator_diffeq, (0.0, 1.0), np.array([1.0, 0.0]), method=name, pass_dy_as_arg=True)
        assert result.success
        assert name[:4].lower() in result.integration_method.lower()


@pytest.mark.parametrize('integration_method', NEW_METHODS)
def test_convergence_order(integration_method):
    """With fixed steps the global error of a smooth non-linear problem falls as h^p."""
    num_steps = {"Tsit5": (64, 128, 256), "Vern7": (32, 64, 128), "Vern8": (16, 32, 64)}[integration_method]
    time_span = (0.0, 10.0)
    y0 = riccati_answer(time_span[0])
    errors = list()
    for steps in num_steps:
        step = (time_span[1] - time_span[0]) / steps
        result = pysolve_ivp(riccati_diffeq, time_span, y0, method=integration_method, rtol=FIXED_STEP_TOL,
                             atol=FIXED_STEP_TOL, first_step=step, max_step=step, pass_dy_as_arg=True)
        assert result.success
        assert abs(result.steps_taken - steps) <= 1
        errors.append(np.max(np.abs(result.y - riccati_answer(result.t))))
    errors = np.asarray(errors)
    observed_orders = np.log2(errors[:-1] / errors[1:])
    expected_order = METHOD_ORDER[integration_method]
    assert np.all(observed_orders > expected_order - 0.6)
    assert np.all(observed_orders < expected_order + 1.0)


@pytest.mark.parametrize('integration_method', NEW_METHODS)
def test_interpolant_order(integration_method):
    """The interpolant's error inside one step from the exact solution falls as h^(q + 1)."""
    steps = (0.4, 0.2, 0.1) if integration_method == "Tsit5" else (0.8, 0.4, 0.2)
    t_start = 0.5
    y0 = riccati_answer(t_start)
    errors = list()
    for step in steps:
        result = pysolve_ivp(riccati_diffeq, (t_start, t_start + 3.0 * step), y0, method=integration_method,
                             rtol=FIXED_STEP_TOL, atol=FIXED_STEP_TOL, first_step=step, max_step=step,
                             dense_output=True, pass_dy_as_arg=True)
        assert result.success
        t_interp = t_start + step * np.linspace(0.05, 0.95, 19)
        errors.append(np.max(np.abs(result(t_interp) - riccati_answer(t_interp))))
    errors = np.asarray(errors)
    observed_orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(observed_orders > INTERPOLANT_ORDER[integration_method] + 1.0 - 0.7)


@pytest.mark.parametrize('integration_method', NEW_METHODS)
@pytest.mark.parametrize('rtol', (1.0e-8, 1.0e-12))
def test_dense_output_matches_step_accuracy(integration_method, rtol):
    """Dense output between the steps is as accurate as the steps themselves, including at tight tolerances where an
    interpolant evaluated from its monomial coefficients would lose digits to cancellation."""
    time_span = (0.0, 6.0 * np.pi)
    result = pysolve_ivp(kepler_diffeq, time_span, kepler_answer(0.0)[:, 0], method=integration_method, rtol=rtol,
                         atol=rtol, dense_output=True, pass_dy_as_arg=True)
    assert result.success
    step_error = np.max(np.abs(result.y - kepler_answer(result.t)))
    t_interp = np.linspace(time_span[0] + 0.01, time_span[1] - 0.01, 2001)
    dense_error = np.max(np.abs(result(t_interp) - kepler_answer(t_interp)))
    assert dense_error < 2.0 * step_error + 1.0e-13
    # Dense output at the stored times reproduces the stored values.
    assert np.allclose(result(result.t), result.y, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize('integration_method', NEW_METHODS)
def test_agrees_with_tight_dop853(integration_method):
    """A tight solution of the non-linear Lotka-Volterra problem agrees with DOP853 at a tighter tolerance."""

    @njit
    def lotka_volterra(dy, t, y):
        dy[0] = (1.0 - 0.01 * y[1]) * y[0]
        dy[1] = (0.02 * y[0] - 1.0) * y[1]

    time_span = (0.0, 20.0)
    y0 = np.array([20.0, 20.0])
    t_eval = np.linspace(0.0, 20.0, 101)
    reference = pysolve_ivp(lotka_volterra, time_span, y0, method="DOP853", t_eval=t_eval, rtol=1.0e-13,
                            atol=1.0e-12, pass_dy_as_arg=True)
    result = pysolve_ivp(lotka_volterra, time_span, y0, method=integration_method, t_eval=t_eval, rtol=1.0e-10,
                         atol=1.0e-10, pass_dy_as_arg=True)
    assert reference.success and result.success
    assert np.allclose(result.y, reference.y, rtol=1.0e-7, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', NEW_METHODS)
@pytest.mark.parametrize('backward', (False, True))
def test_t_eval(integration_method, backward):
    """`t_eval` values come from the interpolant and match the analytic solution in either direction."""
    time_span = (0.0, 10.0)
    t_eval = np.linspace(0.0, 10.0, 37)
    y0 = np.array([1.0, 0.0])
    if backward:
        time_span = (10.0, 0.0)
        t_eval = np.ascontiguousarray(t_eval[::-1])
        y0 = np.array([np.cos(10.0), -np.sin(10.0)])
    result = pysolve_ivp(oscillator_diffeq, time_span, y0, method=integration_method, t_eval=t_eval, rtol=1.0e-10,
                         atol=1.0e-12, pass_dy_as_arg=True)
    assert result.success
    assert np.array_equal(result.t, t_eval)
    assert np.allclose(result.y, np.array([np.cos(t_eval), -np.sin(t_eval)]), rtol=0.0, atol=1.0e-8)


@pytest.mark.parametrize('integration_method', NEW_METHODS)
@pytest.mark.parametrize('terminal', (None, False, 0, True, 1))
def test_events(integration_method, terminal):
    """Event roots are found with the interpolant: y0 = cos(t) crosses zero at pi/2 + k pi. As in SciPy, an unset,
    False, or zero `terminal` never ends the integration, and True or 1 ends it at the first crossing."""

    def crossing(t, y):
        return y[0]

    if terminal is not None:
        crossing.terminal = terminal
    result = pysolve_ivp(oscillator_diffeq, (0.0, 10.0), np.array([1.0, 0.0]), method=integration_method,
                         events=(crossing,), rtol=1.0e-10, atol=1.0e-12, pass_dy_as_arg=True)
    assert result.success
    expected = np.pi / 2.0 + np.pi * np.arange(1 if terminal else 3)
    assert result.t_events[0].size == expected.size
    assert np.allclose(result.t_events[0], expected, rtol=0.0, atol=1.0e-8)
    assert np.allclose(result.y_events[0][1], -np.sin(expected), rtol=0.0, atol=1.0e-8)
    if terminal:
        assert result.event_terminated
        assert np.isclose(result.t[-1], np.pi / 2.0, rtol=0.0, atol=1.0e-8)


def lotka_volterra_py(t, y):
    return np.array([(1.0 - 0.01 * y[1]) * y[0], (0.02 * y[0] - 1.0) * y[1]])


def van_der_pol_py(t, y):
    return np.array([y[1], (1.0 - y[0] ** 2) * y[1] - y[0]])


# Accepted steps and diffeq calls of a reference Python implementation of the same embedded Runge-Kutta step control
# (SciPy's initial step, safety 0.9, factors in [0.2, 10]) with OrdinaryDiffEq.jl's tableaus, for atol = rtol * 1e-3.
REFERENCE_COUNTS = {
    ("lotka", "Tsit5", 1.0e-4): (43, 338),
    ("lotka", "Tsit5", 1.0e-7): (157, 1040),
    ("lotka", "Tsit5", 1.0e-10): (617, 3704),
    ("lotka", "Vern7", 1.0e-4): (34, 522),
    ("lotka", "Vern7", 1.0e-7): (78, 1052),
    ("lotka", "Vern7", 1.0e-10): (189, 2012),
    ("lotka", "Vern8", 1.0e-4): (21, 379),
    ("lotka", "Vern8", 1.0e-7): (42, 743),
    ("lotka", "Vern8", 1.0e-10): (97, 1575),
    ("vdp", "Tsit5", 1.0e-4): (71, 566),
    ("vdp", "Tsit5", 1.0e-7): (261, 1988),
    ("vdp", "Tsit5", 1.0e-10): (1012, 6254),
    ("vdp", "Vern7", 1.0e-4): (54, 742),
    ("vdp", "Vern7", 1.0e-7): (123, 1622),
    ("vdp", "Vern7", 1.0e-10): (309, 3602),
    ("vdp", "Vern8", 1.0e-4): (34, 600),
    ("vdp", "Vern8", 1.0e-7): (73, 1302),
    ("vdp", "Vern8", 1.0e-10): (161, 2537),
    }


@pytest.mark.parametrize('case', list(REFERENCE_COUNTS), ids=lambda case: f"{case[0]}-{case[1]}-{case[2]:.0e}")
def test_step_counts_match_reference(case):
    """The step counts match the reference to within a step or two: a different compiler or floating point contraction
    can flip an accept or reject decision that sits near the threshold. Vern8's error estimate on its first, tiny steps
    is dominated by roundoff in its large coefficients, so its diffeq calls are allowed to drift further."""
    problem_name, integration_method, rtol = case
    diffeq, time_span, y0 = {
        "lotka": (lotka_volterra_py, (0.0, 20.0), np.array([20.0, 20.0])),
        "vdp": (van_der_pol_py, (0.0, 20.0), np.array([2.0, 0.0]))}[problem_name]
    num_calls = [0]

    def counted_diffeq(t, y):
        num_calls[0] += 1
        return diffeq(t, y)

    result = pysolve_ivp(counted_diffeq, time_span, y0, method=integration_method, rtol=rtol, atol=rtol * 1.0e-3)
    assert result.success
    reference_steps, reference_calls = REFERENCE_COUNTS[case]
    call_tolerance = 0.1 if integration_method == "Vern8" else 0.02
    assert abs(result.steps_taken - reference_steps) <= 2
    assert abs(num_calls[0] - reference_calls) <= call_tolerance * reference_calls
