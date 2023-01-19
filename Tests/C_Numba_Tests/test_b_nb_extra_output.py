import numpy as np
from numba import njit

from CyRK import nbrk_ode


@njit
def diffeq_extra_outputs(t, y):
    extra_0 = (1. - 0.01 * y[1])
    extra_1 = (0.02 * y[0] - 1.)
    dy_0 = extra_0 * y[0]
    dy_1 = extra_1 * y[1]
    return np.asarray([dy_0, dy_1, extra_0, extra_1], dtype=y.dtype)


initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8


def test_extra_output_integration():
    """Check that the numba solver is able to run and capture additional outputs """

    time_domain, all_output, success, message = \
        nbrk_ode(diffeq_extra_outputs, time_span, initial_conds, capture_extra=True)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    assert all_output.dtype == np.complex128
    assert time_domain.size > 1
    assert time_domain.size == all_output[0].size
    assert len(all_output.shape) == 2
    assert all_output.shape[0] == 4
    assert all_output[0].size == all_output[1].size
    assert all_output[0].size == all_output[2].size
    assert all_output[0].size == all_output[3].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str


def test_extra_output_integration_teval_no_extra_interpolation():
    """Check that the numba solver is able to run and capture additional outputs.
    Reduced t_eval used but no interpolation used on the extra parameters.
    """

    t_eval = np.linspace(time_span[0], time_span[1], 5)

    time_domain, all_output, success, message = \
        nbrk_ode(
            diffeq_extra_outputs, time_span, initial_conds, t_eval=t_eval,
            capture_extra=True, interpolate_extra=False)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    assert all_output.dtype == np.complex128
    assert time_domain.size == t_eval.size
    assert time_domain.size == all_output[0].size
    assert len(all_output.shape) == 2
    assert all_output.shape[0] == 4
    assert all_output[0].size == all_output[1].size
    assert all_output[0].size == all_output[2].size
    assert all_output[0].size == all_output[3].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str


def test_extra_output_integration_teval_with_extra_interpolation():
    """Check that the numba solver is able to run and capture additional outputs
    Reduced t_eval used with interpolation used on the extra parameters.
    """

    t_eval = np.linspace(time_span[0], time_span[1], 5)

    time_domain, all_output, success, message = \
        nbrk_ode(
                diffeq_extra_outputs, time_span, initial_conds, t_eval=t_eval,
                capture_extra=True, interpolate_extra=True)

    # Check that the ndarrays make sense
    assert type(time_domain) == np.ndarray
    assert time_domain.dtype == np.float64
    assert all_output.dtype == np.complex128
    assert time_domain.size == t_eval.size
    assert time_domain.size == all_output[0].size
    assert len(all_output.shape) == 2
    assert all_output.shape[0] == 4
    assert all_output[0].size == all_output[1].size
    assert all_output[0].size == all_output[2].size
    assert all_output[0].size == all_output[3].size

    # Check that the other output makes sense
    assert type(success) == bool
    assert success
    assert type(message) == str
