import math
import pytest

""" Same tests that SciPy uses for their optimize.brentq. """

# Solve x**3 - A0 = 0  for A0 = [2.0, 2.1, ..., 2.9].
# The ARGS have 3 elements just to show how this could be done for any cubic
# polynomial.
A0_tuple = tuple(-2.0 - x/10.0 for x in range(10))  # constant term
ARGS = (0.0, 0.0, 1.0)  # 1st, 2nd, and 3rd order terms
XLO, XHI = 0.0, 2.0  # first and second bounds of zeros functions
# absolute and relative tolerances and max iterations for zeros functions
XTOL, RTOL, MITR = 0.001, 0.001, 10
EXPECTED_list = [(-a0) ** (1.0/3.0) for a0 in A0_tuple]
# = [1.2599210498948732,
#    1.2805791649874942,
#    1.300591446851387,
#    1.3200061217959123,
#    1.338865900164339,
#    1.3572088082974532,
#    1.375068867074141,
#    1.3924766500838337,
#    1.4094597464129783,
#    1.4260431471424087]


@pytest.mark.parametrize('A0_index', range(len(A0_tuple)))
def test_brentq(A0_index):
    from CyRK.optimize.optimize_test import brentq_test

    A0 = A0_tuple[A0_index]
    expected = EXPECTED_list[A0_index]
    result = brentq_test(A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)
    assert math.isclose(result, expected, rel_tol=RTOL, abs_tol=XTOL)
