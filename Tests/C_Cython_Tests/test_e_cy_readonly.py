import numpy as np
from numba import njit
import pytest

from CyRK import cyrk_ode
from CyRK.cy.cysolvertest import CySolverTester

@njit
def diffeq(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]


@njit
def diffeq_args(t, y, dy, a, b):
    dy[0] = (1. - a * y[1]) * y[0]
    dy[1] = (b * y[0] - 1.) * y[1]

initial_conds = np.asarray((20., 20.), dtype=np.float64)
initial_conds_complex = np.asarray((20. + 0.01j, 20. - 0.01j), dtype=np.complex128)
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8

@pytest.mark.parametrize('complex_valued', (True, False))
def test_readonly_y0(complex_valued):
    """ Test if a readonly array will work in cyrk_ode. """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    # Make readonly
    initial_conds_to_use.setflags(write=False)

    time_domain, y_results, success, message = \
        cyrk_ode(diffeq, time_span, initial_conds_to_use, rk_method=1)

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
def test_readonly_y0_CySolver(complex_valued):
    """ Test if a readonly array will work in CySolver. """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    # Make readonly
    initial_conds_to_use.setflags(write=False)


    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=1, auto_solve=True)

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