"""Tests for CyRK's implicit integrators: BDF, LSODA, and Radau.

The problems used here are stiff on purpose: they are cheap for an implicit method but force an
explicit method to take very small steps, which is what makes the comparisons meaningful.
"""

import numpy as np
import pytest
from numba import njit

from CyRK import pysolve_ivp, ODEMethod
from CyRK.cy.cysolver_test import cytester
from CyRK.cy.pyhelpers import find_ode_method_int

IMPLICIT_METHODS = ("BDF", "LSODA", "RADAU")

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
    assert int(ODEMethod.RADAU) == 8
    assert find_ode_method_int('bdf') == int(ODEMethod.BDF)
    assert find_ode_method_int('LSODA') == int(ODEMethod.LSODA)
    assert find_ode_method_int('Radau') == int(ODEMethod.RADAU)


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_accuracy(integration_method):
    """Every implicit method should reproduce the exact solution of a stiff problem."""
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


CHAIN_COUPLING = 2.0e3
CHAIN_MATRIX = np.array([[-1.0,           0.0,            0.0,            0.0],
                         [CHAIN_COUPLING, -2.0,           0.0,            0.0],
                         [0.0,            CHAIN_COUPLING, -3.0,           0.0],
                         [0.0,            0.0,            CHAIN_COUPLING, -1.0e4]])


@njit
def chain_diffeq(dy, t, y):
    """Stiff linear chain in which each variable drives the next far harder than it decays, so the
    factorization of the Newton iteration matrix pivots in three of its four columns."""
    dy[0] = -y[0]
    dy[1] = CHAIN_COUPLING * y[0] - 2.0 * y[1]
    dy[2] = CHAIN_COUPLING * y[1] - 3.0 * y[2]
    dy[3] = CHAIN_COUPLING * y[2] - 1.0e4 * y[3]


@pytest.mark.parametrize('integration_method', IMPLICIT_METHODS)
def test_implicit_newton_solve_with_pivoting(integration_method):
    """A Newton iteration matrix that needs row interchanges must still be solved exactly. Before
    v0.19.5 the dense LU solve applied the interchanges in the wrong order, so BDF and Radau took
    over 6000 steps on this problem (SciPy takes about 200) with corrections that were wrong."""
    from scipy.linalg import expm

    y0 = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    result = pysolve_ivp(chain_diffeq, (0.0, 10.0), y0, method=integration_method,
                         rtol=1.0e-6, atol=1.0e-10, pass_dy_as_arg=True)
    exact = expm(10.0 * CHAIN_MATRIX) @ y0

    assert result.success
    assert result.steps_taken < 500
    assert np.allclose(result.y[:, -1], exact, rtol=1.0e-3, atol=1.0e-6 * np.max(np.abs(exact)))


@njit
def constant_diffeq(dy, t, y):
    """Nothing changes, so every error estimate is exactly zero."""
    dy[0] = 0.0
    dy[1] = 0.0


def test_radau_step_growth_at_zero_error():
    """A zero error estimate must grow Radau's step by the full `MAX_FACTOR` (10), as in SciPy. Before v0.19.5 it grew
    by the safety factor times 10 and took 18 steps here where SciPy takes 17."""
    from scipy.integrate import solve_ivp

    y0 = np.asarray((1.0, 2.0), dtype=np.float64)
    result = pysolve_ivp(constant_diffeq, (0.0, 1.0e10), y0, method='Radau', pass_dy_as_arg=True)
    reference = solve_ivp(lambda t, y: np.zeros(2), (0.0, 1.0e10), y0, method='Radau')

    assert result.success
    assert result.size == reference.t.size


A_ROBERTSON = np.asarray((0.04, 1.0e4, 3.0e7), dtype=np.float64)


def robertson_scipy(t, y):
    a, b, c = A_ROBERTSON
    return np.asarray((-a * y[0] + b * y[1] * y[2], a * y[0] - b * y[1] * y[2] - c * y[1] * y[1], c * y[1] * y[1]))


def robertson_scipy_jacobian(t, y):
    a, b, c = A_ROBERTSON
    return np.asarray(((-a, b * y[2], b * y[1]),
                       (a, -b * y[2] - 2.0 * c * y[1], -b * y[1]),
                       (0.0, 2.0 * c * y[1], 0.0)))


@pytest.mark.parametrize('integration_method', ("BDF", "LSODA", "Radau"))
def test_implicit_analytic_jacobian_matches_scipy(integration_method):
    """With the same analytic Jacobian, `cysolve_ivp` (through `jac_ptr`) and SciPy run the same algorithm, so they
    take the same steps up to round-off and reach the same answer."""
    from scipy.integrate import solve_ivp

    y0 = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    result = cytester(11, (0.0, 1.0e5), y0, args=A_ROBERTSON, method=integration_method, rtol=1.0e-6, atol=1.0e-10,
                      use_jacobian=True)
    reference = solve_ivp(robertson_scipy, (0.0, 1.0e5), y0, method=integration_method, rtol=1.0e-6, atol=1.0e-10,
                          jac=robertson_scipy_jacobian)

    assert result.success
    assert reference.success
    assert abs(result.size - reference.t.size) <= 0.05 * reference.t.size
    assert np.allclose(result.y[:, -1], reference.y[:, -1], rtol=1.0e-4, atol=1.0e-9)


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


@njit
def banded_chain_diffeq(dy, t, y, feedback):
    """Stiff chain whose variables each drive the next far harder than they decay, plus an optional weak negative
    feedback from the next variable. Its Jacobian is lower bidiagonal or tridiagonal and its band factorization
    pivots. The last variable decays fastest."""
    num_y = y.size
    dy[0] = -y[0]
    for i in range(1, num_y):
        dy[i] = 2.0e3 * y[i - 1] - (i + 1.0) * y[i]
    for i in range(num_y - 1):
        dy[i] += feedback * y[i + 1]
    dy[num_y - 1] -= (1.0e4 - num_y) * y[num_y - 1]


@pytest.mark.parametrize('lband, uband, feedback', ((1, 0, 0.0), (1, 1, -0.01), (2, 1, -0.01)))
def test_lsoda_banded_jacobian_with_pivoting(lband, uband, feedback):
    """The banded LU factorization must clear the fill-in rows left by the previous factorization, as LAPACK's
    dgbtf2 does; LSODA's finite-difference Jacobian only rewrites the band. Before v0.19.5 the stale fill-in entered
    U through the row interchanges, and with `lband=1, uband=0` LSODA failed after 310 steps (dense: 492)."""
    from scipy.linalg import expm

    num_y = 8
    matrix = np.diag(-np.arange(1.0, num_y + 1.0)) + np.diag(np.full(num_y - 1, 2.0e3), -1) + \
        np.diag(np.full(num_y - 1, feedback), 1)
    matrix[-1, -1] = -1.0e4
    y0 = np.zeros(num_y, dtype=np.float64)
    y0[0] = 1.0
    exact = expm(10.0 * matrix) @ y0

    dense_result = pysolve_ivp(banded_chain_diffeq, (0.0, 10.0), y0, method='LSODA', args=(feedback,),
                               rtol=1.0e-6, atol=1.0e-10, pass_dy_as_arg=True)
    banded_result = pysolve_ivp(banded_chain_diffeq, (0.0, 10.0), y0, method='LSODA', args=(feedback,),
                                rtol=1.0e-6, atol=1.0e-10, lband=lband, uband=uband, pass_dy_as_arg=True)

    assert dense_result.success
    assert banded_result.success
    # The two factorizations differ only in round-off, so the steps and the answers stay close.
    assert abs(banded_result.size - dense_result.size) < 0.1 * dense_result.size
    assert np.allclose(banded_result.y[:, -1], dense_result.y[:, -1], rtol=1.0e-4, atol=1.0e-6 * np.max(np.abs(exact)))
    # LSODA's global error on this oscillating chain is a few percent (SciPy's is the same).
    assert np.allclose(banded_result.y[:, -1], exact, rtol=5.0e-2, atol=1.0e-3 * np.max(np.abs(exact)))


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


@njit
def fading_stiffness_diffeq(dy, t, y, t_start, stiffness):
    """Stiff at first, with a stiffness that fades away; time is measured from `t_start`."""
    tau = t - t_start
    rate = stiffness * np.exp(-tau) + 1.0
    dy[0] = -rate * (y[0] - np.cos(tau))
    dy[1] = y[0] - 0.1 * y[1]


@pytest.mark.parametrize('t_start, stiffness', ((3.0e8, 1.0e4), (3.0e8, 1.0e5), (1.0e8, 1.0e6)))
def test_lsoda_survives_steps_at_the_spacing_of_t(t_start, stiffness):
    """LSODA must go on through a few steps no larger than the spacing between numbers at t, as ODEPACK does.

    Far from t = 0 (the spacing is 1.5e-8 at t = 1e8 and 6e-8 at t = 3e8), error test failures where cos(tau) passes
    zero cut LSODA's step to the spacing for a few steps before it grows back. CyRK v0.19.2 to v0.19.4 failed at the
    first such step in each of these cases. Which steps land there depends on round-off, so a platform may not reach
    the spacing at all; the integration must succeed either way.
    """
    rtol = 1.0e-8
    atol = rtol * 1.0e-3
    y0 = np.asarray((1.0, 0.0), dtype=np.float64, order='C')
    result = pysolve_ivp(fading_stiffness_diffeq, (t_start, t_start + 20.0), y0, method='LSODA',
                         args=(t_start, stiffness), rtol=rtol, atol=atol, pass_dy_as_arg=True)
    # The same problem near t = 0, where the spacing between numbers is no concern.
    reference = pysolve_ivp(fading_stiffness_diffeq, (0.0, 20.0), y0, method='LSODA',
                            args=(0.0, stiffness), rtol=1.0e-10, atol=1.0e-13, pass_dy_as_arg=True)

    assert result.success
    assert result.t[-1] == t_start + 20.0
    assert reference.success
    # Agreement is limited by how finely t itself is resolved so far from zero (errors of a few 1e-6 at t = 3e8,
    # with or without the steps at the spacing).
    assert np.allclose(result.y[:, -1], reference.y[:, -1], rtol=1.0e-4, atol=1.0e-5)


@njit
def slow_manifold_diffeq(dy, t, y, stiffness):
    """A fast variable relaxing onto 1e-6 times a slowly decaying one."""
    dy[0] = -stiffness * (y[0] - 1.0e-6 * y[1])
    dy[1] = -1.0e-3 * y[1]


@pytest.mark.parametrize('stiffness', (1.0e6, 1.0e9, 1.0e12, 1.0e15))
@pytest.mark.parametrize('rtol', (1.0e-3, 1.0e-6))
def test_lsoda_starts_on_a_stiff_slow_manifold(stiffness, rtol):
    """LSODA must start a stiff problem whose initial state sits on its slow manifold.

    There f(t0) has no fast component, so the first step size, taken from f(t0), is many times what the Adams
    functional iteration can converge at. ODEPACK (and SciPy's LSODA) gave up after cutting it by 4 ten times and
    failed every case here with automatic first steps but stiffness 1e6 at rtol 1e-6.
    """
    y0 = np.asarray((1.0e-6, 1.0), dtype=np.float64, order='C')
    atol = np.asarray((1.0e-14, 1.0e-10), dtype=np.float64, order='C')
    result = pysolve_ivp(slow_manifold_diffeq, (0.0, 1000.0), y0, method='LSODA', args=(stiffness,),
                         rtol=rtol, atol=atol, pass_dy_as_arg=True)

    assert result.success, result.message
    # The method switches to BDF after its first stiffness test, so the stiffness costs few steps.
    assert result.steps_taken < 150
    slow_exact = np.exp(-1.0)
    assert np.isclose(result.y[1, -1], slow_exact, rtol=20.0 * rtol)
    assert np.isclose(result.y[0, -1], 1.0e-6 * slow_exact, rtol=20.0 * rtol)


@njit
def gravity_center_diffeq(dy, r, y):
    """Gravity, pressure, mass, and moment of inertia of a uniform sphere (G = 1 / pi, rho = 1), from its center."""
    if r == 0.0:
        # The removable 1/r singularity of the gravity equation, at its limit.
        dy[0] = 4.0 / 3.0
        dy[1] = 0.0
        dy[2] = 0.0
        dy[3] = 0.0
    else:
        dy[0] = 4.0 - 2.0 * y[0] / r
        dy[1] = -y[0]
        dy[2] = 4.0 * np.pi * r * r
        dy[3] = (2.0 / 3.0) * dy[2] * r * r


@pytest.mark.parametrize('diffeq, time_span, y0, rtol, atol, args, expected', (
    # A stiff problem whose first step converged within ODEPACK's retries (see the test above).
    (slow_manifold_diffeq, (0.0, 1.0e4), (1.0e-6, 1.0), 1.0e-9, (1.0e-14, 1.0e-10), (1.0e6,),
     (1.0e-6 * np.exp(-10.0), np.exp(-10.0))),
    # A stiffness of 2 / r near the singular center of a planet's structure integration.
    (gravity_center_diffeq, (0.0, 1.0), (0.0, 2.0 / 3.0, 0.0, 0.0), 1.0e-4, 1.0e-20, (),
     (4.0 / 3.0, 0.0, 4.0 * np.pi / 3.0, 8.0 * np.pi / 15.0)),
))
def test_lsoda_leaves_a_step_held_at_the_adams_stability_bound(diffeq, time_span, y0, rtol, atol, args, expected):
    """LSODA must switch to BDF rather than hold the step at the Adams stability bound without end.

    In both problems the Adams corrector converges at roundoff, which leaves no fresh Lipschitz estimate, while the
    step is held at the stability bound of the one it has. ODEPACK (and SciPy's LSODA) held these steps at 5.3e-7 and
    7.5e-12 for as long as they were let run.
    """
    y0 = np.asarray(y0, dtype=np.float64, order='C')
    atol = np.asarray(atol, dtype=np.float64, order='C') if isinstance(atol, tuple) else atol
    result = pysolve_ivp(diffeq, time_span, y0, method='LSODA', args=args, rtol=rtol, atol=atol,
                         pass_dy_as_arg=True, max_num_steps=5000)

    assert result.success, result.message
    assert result.steps_taken < 500
    assert np.allclose(result.y[:, -1], expected, rtol=1.0e-3, atol=1.0e-6)


@pytest.mark.parametrize('integration_method', ("RK23", "RK45", "DOP853", "Tsit5", "Vern7", "Vern8", "BDF", "RADAU"))
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
