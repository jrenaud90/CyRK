import pytest
import numpy as np

from CyRK.cy.cysolvertest import CySolverTester

initial_conds = np.asarray((20., 20.), dtype=np.float64)
initial_conds_complex = np.asarray((20. + 0.01j, 20. - 0.01j), dtype=np.complex128)
time_span = (0., 10.)
time_span_large = (0., 1000.)
rtol = 1.0e-7
atol = 1.0e-8


@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_CySolverTester_resolve(rk_method, complex_valued):
    """Check that the cython class solver produced the correct result if it is re-solved multiple times"""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method, auto_solve=False)

    # Solve once
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success
    solution_1_t = np.copy(CySolverTesterInst.t)
    solution_1_y = np.copy(CySolverTesterInst.y)

    # Solve twice
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success
    solution_2_t = np.copy(CySolverTesterInst.t)
    solution_2_y = np.copy(CySolverTesterInst.y)

    # Solve thrice
    CySolverTesterInst.solve()
    assert CySolverTesterInst.success
    solution_3_t = np.copy(CySolverTesterInst.t)
    solution_3_y = np.copy(CySolverTesterInst.y)

    # Check outputs
    # 1 and 2
    assert solution_1_t.shape == solution_2_t.shape
    assert np.all(solution_1_t == solution_2_t)
    assert solution_1_y.shape == solution_2_y.shape
    assert np.all(solution_1_y == solution_2_y)

    # 2 and 3 (if the above passed then this will implicitly check 1 and 3 too).
    assert solution_2_t.shape == solution_3_t.shape
    assert np.all(solution_2_t == solution_3_t)
    assert solution_2_y.shape == solution_3_y.shape
    assert np.all(solution_2_y == solution_3_y)


@pytest.mark.parametrize('complex_valued', (False,))
@pytest.mark.parametrize('rk_method', (0, 1, 2))
def test_CySolverTester_multi_resolve(rk_method, complex_valued):
    """Check that the cython class solver can be resolved many times without memory access issues."""

    if complex_valued:
        initial_conds_to_use = initial_conds_complex
    else:
        initial_conds_to_use = initial_conds

    CySolverTesterInst = CySolverTester(time_span, initial_conds_to_use, rk_method=rk_method, auto_solve=False)

    for i in range(10_000):
        # Change y0 values for each loop
        y0 = np.copy(initial_conds_to_use)
        y0[0] += 0.1
        y0[1] -= 0.1
        CySolverTesterInst.change_y0(y0, auto_reset_state=False)
        CySolverTesterInst.solve(reset=True)
        assert CySolverTesterInst.success
