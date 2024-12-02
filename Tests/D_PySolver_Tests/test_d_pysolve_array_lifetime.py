import numpy as np
from CyRK import pysolve_ivp
import gc


def diffeq(dy, t, y):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]

initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-4
atol = 1.0e-7

def run_diffeq_array_return():
    result = pysolve_ivp(diffeq, time_span, initial_conds, method="RK45", rtol=rtol, atol=atol, pass_dy_as_arg=True)

    t = result.t
    y = result.y
    del result
    gc.collect()
    return t, y

def run_diffeq_result_return():
    result = pysolve_ivp(diffeq, time_span, initial_conds, method="RK45", rtol=rtol, atol=atol, pass_dy_as_arg=True)
    return result

def test_pysolve_array_lifetimes():
    """ Tests that the underlying arrays returned by the WrapCySolverResult class can outlive the class instance. """
    
    # Run in a loop because sometimes memory corruptions can take a while to appear
    for i in range(50):
        # Get result from the result instance class
        result = run_diffeq_result_return()

        # Get arrays from the array return
        t, y = run_diffeq_array_return()

        # Check that they contain the same data
        assert result.y.size == y.size
        assert result.t.size == t.size

        assert np.allclose(result.y, y)
        assert np.allclose(result.t, t)
