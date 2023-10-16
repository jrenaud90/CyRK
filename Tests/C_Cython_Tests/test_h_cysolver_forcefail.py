import pytest
import numpy as np

from CyRK.cy.cysolvertest import CySolverTester


initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
initial_conds_complex = np.asarray((20. + 0.01j, 20. - 0.01j), dtype=np.complex128, order='C')
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8

rtols = np.asarray((1.0e-7, 1.0e-8), dtype=np.float64, order='C')
atols = np.asarray((1.0e-8, 1.0e-9), dtype=np.float64, order='C')


def wrapper_func(initial_conds_to_use, rtols_use, atols_use, rk_method):
    
    # Build solver instance in a wrapper function
    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use,
                                        rtol=rtol, atol=atol, rtols=rtols_use, atols=atols_use,
                                        rk_method=rk_method, auto_solve=True, force_fail=True)
    
    result = CySolverTesterInst.success

    # Delete solver
    del CySolverTesterInst

    return result


@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
@pytest.mark.parametrize('use_rtol_array', (True, False))
@pytest.mark.parametrize('use_atol_array', (True, False))
def test_basic_forcefailing_CySolverTester(use_atol_array, use_rtol_array, rk_method, complex_valued):
    """Check that the cython class solver is able to run with its default arguments"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    if use_atol_array:
        atols_use = atols
    else:
        atols_use = None
    if use_rtol_array:
        rtols_use = rtols
    else:
        rtols_use = None

    result = wrapper_func(initial_conds_to_use, rtols_use, atols_use, rk_method)

    assert not result
