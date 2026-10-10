import numpy as np
import pytest
from numba import njit

from CyRK import pysolve_ivp

def diffeq(dy, t, y,):
    # Real dy
    dy[0] = 3.1 * t - y[1]
    dy[1] = y[0] * (0.3 * t * y[1])
    # Extra output
    dy[2] = 0.25
    dy[3] = t / 2.

t_span = (0.0, 10.0)
y0 = np.asarray([5., 2.], dtype=np.float64)

def test_pysolve_extra_output_with_dense():
    """ Test that pysolve (and by extension, cysolve) can interpolate extra outputs when `dense_output = True` """

    result = pysolve_ivp(
        diffeq,
        t_span,
        y0,
        method = 'RK45',
        t_eval = None,
        dense_output = True,
        args = None,
        expected_size = 0,
        num_extra = 2,
        first_step = 0.0,
        max_step = 100_000,
        rtol = 1.0e-3,
        atol = 1.0e-6,
        max_num_steps = 0,
        max_ram_MB = 2000,
        pass_dy_as_arg = True
        )

    assert result.success

    # Call dense output with a float
    dense_out_float = result(0.3)
    expected_float = np.asarray([[4.52627329], [2.13093483], [0.25      ], [0.15      ]], dtype=np.float64)
    assert dense_out_float.shape == (4, 1)
    assert np.allclose(dense_out_float, expected_float)

    # Call dense output with an array
    dense_out_array = result(np.asarray([0.3, 4.0, 8., 9.10, 9.9]))
    expected_array = np.asarray([[ 4.52627329,  1.87160845,  0.13355624,  1.57871591, -1.01590877],
                                 [ 2.13093483, 21.05384256, 10.31201731, 48.55169176, 57.06848508],
                                 [ 0.25,        0.25,        0.25,        0.25,        0.25      ],
                                 [ 0.15,        2.,          4.,          4.55,        4.95      ]], dtype=np.float64)
    assert dense_out_array.shape == (4, 5)
    assert np.allclose(dense_out_array, expected_array)


ALL_METHODS = ("RK23", "RK45", "DOP853", "Tsit5", "Vern7", "Vern8", "BDF", "LSODA", "Radau")


@pytest.mark.parametrize('integration_method', ALL_METHODS)
@pytest.mark.parametrize('time_span', ((0.0, 6.0), (6.0, 0.0)), ids=("forward", "backward"))
def test_dense_output_uses_the_right_step(integration_method, time_span):
    """Dense output uses the interpolant of the step that contains t, in either direction: at t_start, inside steps,
    at the stored step times, and at t_end it is finite, reproduces the stored values, and is about as accurate as the
    steps themselves."""

    @njit
    def nonlinear_diffeq(dy, t, y):
        dy[0] = -2.0 * t * y[0] * y[0]
        dy[1] = y[1] * np.cos(t)

    def answer(t):
        return np.array([1.0 / (1.0 + t * t), np.exp(np.sin(t))])

    result = pysolve_ivp(nonlinear_diffeq, time_span, answer(time_span[0]), method=integration_method,
                         rtol=1.0e-10, atol=1.0e-12, dense_output=True, pass_dy_as_arg=True)
    assert result.success
    step_error = np.max(np.abs(result.y - answer(result.t)))
    midpoints = 0.5 * (result.t[:-1] + result.t[1:])
    t_check = np.concatenate((midpoints, np.linspace(0.0, 6.0, 61)))
    y_check = result(t_check)
    assert np.all(np.isfinite(y_check))
    assert np.allclose(result(result.t), result.y, rtol=1.0e-12, atol=1.0e-14)
    # Several interpolants are one order below their method, so they can be somewhat less accurate than the steps; an
    # interpolant from the wrong step is wrong at the 1e-5 level or worse.
    assert np.max(np.abs(y_check - answer(t_check))) < 20.0 * step_error + 1.0e-9
