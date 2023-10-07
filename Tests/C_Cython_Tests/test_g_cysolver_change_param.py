import pytest
import numpy as np

from CyRK.cy.cysolvertest import CySolverTester

initial_conds = np.asarray((20., 20.), dtype=np.float64)
initial_conds_2 = np.asarray((-10., 10.), dtype=np.float64)
initial_conds_complex = np.asarray((20. + 0.01j, 20. - 0.01j), dtype=np.complex128)
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8

rtols = np.asarray((1.0e-7, 1.0e-8), dtype=np.float64, order='C')
atols = np.asarray((1.0e-8, 1.0e-9), dtype=np.float64, order='C')

@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_CySolverTester_change_param(rk_method, complex_valued):
    """ Test CySolver's change parameters functionality. """

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method, auto_solve=False,
                                        rtol=1.0e-8, atol=1.0e-9)

    # Solve once
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success
    solution_1_t = np.copy(CySolverTesterInst.t)
    solution_1_y = np.copy(CySolverTesterInst.y)
    assert solution_1_t[0] == 0.
    assert solution_1_t[-1] == 10.

    # Change timespan and solve again
    CySolverTesterInst.change_t_span((0., 1.))
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success
    solution_2_t = np.copy(CySolverTesterInst.t)
    solution_2_y = np.copy(CySolverTesterInst.y)
    assert solution_2_t[0] == 0.
    assert solution_2_t[-1] == 1.

    # Change several things at once but keep the previous time span the same.
    CySolverTesterInst.change_parameters(rtol=1.0e-11, atol=1.0e-12)
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success
    solution_3_t = np.copy(CySolverTesterInst.t)
    solution_3_y = np.copy(CySolverTesterInst.y)
    assert solution_3_t[0] == 0.
    assert solution_3_t[-1] == 1.
    # Due to the lower tolerances, we expect this solution to be larger than the previous ones.
    assert solution_3_t.size > solution_2_t.size

    # Check that changing teval works
    t_eval = np.linspace(0., 0.5, 10)
    CySolverTesterInst.change_t_eval(t_eval)
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success
    solution_4_t = np.copy(CySolverTesterInst.t)
    solution_4_y = np.copy(CySolverTesterInst.y)
    assert solution_4_t[0] == 0.
    assert solution_4_t[-1] == 0.5
    assert solution_4_t.size == 10
    assert solution_4_y.shape == (2, 10)

    # Check changing rtols/atols array
    CySolverTesterInst.change_parameters(rtols=rtols, atols=atols)
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success

    # Check that the correct error is raised when an incorrect y0 is provided
    y0_bad = np.asarray((-10., 0., 10.), dtype=np.float64)  # Should only have 2 values
    with pytest.raises(AttributeError):
        CySolverTesterInst.change_y0(y0_bad)

    # Check again with the wrapper change function
    with pytest.raises(AttributeError):
        CySolverTesterInst.change_parameters(y0=y0_bad)

    # Check changing rtol/atol to a bad array
    # Too many rtols and atols
    bad_rtols = np.asarray((1.0e-6, 1.0e-7, 1.0e-8), dtype=np.float64, order='C')
    bad_atols = np.asarray((1.0e-7, 1.0e-8, 1.0e-9), dtype=np.float64, order='C')
    # Check again with the wrapper change function
    with pytest.raises(AttributeError):
        CySolverTesterInst.change_parameters(rtols=bad_rtols)
    with pytest.raises(AttributeError):
        CySolverTesterInst.change_parameters(atols=bad_atols)
