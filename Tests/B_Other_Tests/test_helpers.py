import numpy as np
from numba import njit

from CyRK import nb2cy, cy2nb, nbsolve_ivp, pysolve_ivp


@njit
def diffeq_cy(dy, t, y):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]


@njit
def diffeq_cy_args(dy, t, y, a, b):
    dy[0] = (1. - a * y[1]) * y[0]
    dy[1] = (b * y[0] - 1.) * y[1]


@njit
def diffeq_scipy(t, y):
    dy = np.zeros_like(y)
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy


@njit
def diffeq_scipy_args(t, y, a, b):
    dy = np.zeros_like(y)
    dy[0] = (1. - a * y[1]) * y[0]
    dy[1] = (b * y[0] - 1.) * y[1]
    return dy


initial_conds = np.asarray((20., 20.), dtype=np.float64)
time_span = (0., 20.)
rtol = 1.0e-7
atol = 1.0e-8
t_eval = np.linspace(time_span[0], time_span[1], 10)


def test_nb2cy_noargs():
    """ Test converting a scipy diffeq into a cyrk one with no additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    result_nb = \
        nbsolve_ivp(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert result_nb.success

    # Perform a cyrk integration using the diffeq that was written for cyrk
    result_cy = \
        pysolve_ivp(diffeq_cy, time_span, initial_conds,
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)

    assert result_cy.success

    # Perform the function conversion
    diffeq_cy_converted = nb2cy(diffeq_scipy)

    # Use this function to recalculate using cyrk
    result_cy_conv = \
        pysolve_ivp(diffeq_cy_converted, time_span, initial_conds,
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy_conv.success

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((result_cy_conv.y[0] - result_cy.y[0]) / (result_cy_conv.y[0] + result_cy.y[0]))
    p_diff_2_cy = 2. * np.abs((result_cy_conv.y[1] - result_cy.y[1]) / (result_cy_conv.y[1] + result_cy.y[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((result_cy_conv.y[0] - result_nb.y[0]) / (result_cy_conv.y[0] + result_nb.y[0]))
    p_diff_2_nb = 2. * np.abs((result_cy_conv.y[1] - result_nb.y[1]) / (result_cy_conv.y[1] + result_nb.y[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_nb2cy_args():
    """ Test converting a scipy diffeq into a cyrk one with additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    result_nb = \
        nbsolve_ivp(diffeq_scipy_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert result_nb.success

    # Perform a cyrk integration using the diffeq that was written for cyrk
    result_cy = \
        pysolve_ivp(diffeq_cy_args, time_span, initial_conds,
                    args=(0.01, 0.02),
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy.success

    # Perform the function conversion
    diffeq_cy_converted_args = nb2cy(diffeq_scipy_args)

    # Use this function to recalculate using cyrk
    result_cy_conv = \
        pysolve_ivp(diffeq_cy_converted_args, time_span, initial_conds,
                    args=(0.01, 0.02),
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy_conv.success

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((result_cy_conv.y[0] - result_cy.y[0]) / (result_cy_conv.y[0] + result_cy.y[0]))
    p_diff_2_cy = 2. * np.abs((result_cy_conv.y[1] - result_cy.y[1]) / (result_cy_conv.y[1] + result_cy.y[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((result_cy_conv.y[0] - result_nb.y[0]) / (result_cy_conv.y[0] + result_nb.y[0]))
    p_diff_2_nb = 2. * np.abs((result_cy_conv.y[1] - result_nb.y[1]) / (result_cy_conv.y[1] + result_nb.y[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_cy2nb_noargs():
    """ Test converting a cyrk diffeq into a numba/scipy one with no additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    result_nb = \
        nbsolve_ivp(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert result_nb.success

    # Perform a cyrk integration using the diffeq that was written for cyrk
    result_cy = \
        pysolve_ivp(diffeq_cy, time_span, initial_conds,
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy.success

    # Perform the function conversion
    diffeq_nb_converted = cy2nb(diffeq_cy)

    # Use this function to recalculate using cyrk
    result_nb_conv = \
        nbsolve_ivp(diffeq_nb_converted, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert result_nb.success

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((result_nb_conv.y[0] - result_cy.y[0]) / (result_nb_conv.y[0] + result_cy.y[0]))
    p_diff_2_cy = 2. * np.abs((result_nb_conv.y[1] - result_cy.y[1]) / (result_nb_conv.y[1] + result_cy.y[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((result_nb_conv.y[0] - result_nb.y[0]) / (result_nb_conv.y[0] + result_nb.y[0]))
    p_diff_2_nb = 2. * np.abs((result_nb_conv.y[1] - result_nb.y[1]) / (result_nb_conv.y[1] + result_nb.y[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_cy2nb_args():
    """ Test converting a cyrk diffeq into a numba/scipy one with additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    result_nb = \
        nbsolve_ivp(diffeq_scipy_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert result_nb.success

    # Perform a cyrk integration using the diffeq that was written for cyrk
    result_cy = \
        pysolve_ivp(diffeq_cy_args, time_span, initial_conds,
                    args=(0.01, 0.02),
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy.success

    # Perform the function conversion
    diffeq_nb_converted_args = cy2nb(diffeq_cy_args)

    # Use this function to recalculate using cyrk
    nb_result_conv = \
        nbsolve_ivp(diffeq_nb_converted_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert nb_result_conv.success

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((nb_result_conv.y[0] - result_cy.y[0]) / (nb_result_conv.y[0] + result_cy.y[0]))
    p_diff_2_cy = 2. * np.abs((nb_result_conv.y[1] - result_cy.y[1]) / (nb_result_conv.y[1] + result_cy.y[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((nb_result_conv.y[0] - result_nb.y[0]) / (nb_result_conv.y[0] + result_nb.y[0]))
    p_diff_2_nb = 2. * np.abs((nb_result_conv.y[1] - result_nb.y[1]) / (nb_result_conv.y[1] + result_nb.y[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_cy2nb_cache_njit():
    """ Test converting a cyrk diffeq into a numba/scipy with njit cacheing on. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    result_nb = \
        nbsolve_ivp(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert result_nb.success

    # Perform a cyrk integration using the diffeq that was written for cyrk
    result_cy = \
        pysolve_ivp(diffeq_cy, time_span, initial_conds,
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy.success

    # Perform the function conversion
    diffeq_nb_converted = cy2nb(diffeq_cy)

    # Use this function to recalculate using cyrk
    nb_result_conv = \
        nbsolve_ivp(diffeq_nb_converted, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert nb_result_conv.success

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((nb_result_conv.y[0] - result_cy.y[0]) / (nb_result_conv.y[0] + result_cy.y[0]))
    p_diff_2_cy = 2. * np.abs((nb_result_conv.y[1] - result_cy.y[1]) / (nb_result_conv.y[1] + result_cy.y[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((nb_result_conv.y[0] - result_nb.y[0]) / (nb_result_conv.y[0] + result_nb.y[0]))
    p_diff_2_nb = 2. * np.abs((nb_result_conv.y[1] - result_nb.y[1]) / (nb_result_conv.y[1] + result_nb.y[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_nb2cy_cache_njit():
    """ Test converting a scipy diffeq into a cyrk with njit cacheing on. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    result_nb = \
        nbsolve_ivp(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert result_nb.success

    # Perform a cyrk integration using the diffeq that was written for cyrk
    result_cy = \
        pysolve_ivp(diffeq_cy, time_span, initial_conds,
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy.success

    # Perform the function conversion
    diffeq_cy_converted = nb2cy(diffeq_scipy)

    # Use this function to recalculate using cyrk
    result_cy_conv = \
        pysolve_ivp(diffeq_cy_converted, time_span, initial_conds,
                    rtol=rtol, atol=atol, t_eval=t_eval,
                    pass_dy_as_arg=True)
    assert result_cy_conv.success

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((result_cy_conv.y[0] - result_cy.y[0]) / (result_cy_conv.y[0] + result_cy.y[0]))
    p_diff_2_cy = 2. * np.abs((result_cy_conv.y[1] - result_cy.y[1]) / (result_cy_conv.y[1] + result_cy.y[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((result_cy_conv.y[0] - result_nb.y[0]) / (result_cy_conv.y[0] + result_nb.y[0]))
    p_diff_2_nb = 2. * np.abs((result_cy_conv.y[1] - result_nb.y[1]) / (result_cy_conv.y[1] + result_nb.y[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)
