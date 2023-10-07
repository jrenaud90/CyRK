import numpy as np
from numba import njit

from CyRK import nb2cy, cy2nb, nbrk_ode, cyrk_ode


@njit
def diffeq_cy(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]


@njit
def diffeq_cy_args(t, y, dy, a, b):
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


initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 20.)
rtol = 1.0e-7
atol = 1.0e-8
t_eval = np.linspace(time_span[0], time_span[1], 10)


def test_nb2cy_noargs():
    """ Test converting a scipy diffeq into a cyrk one with no additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    time_domain_nb, y_results_nb, success_nb, message_nb = \
        nbrk_ode(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_nb

    # Perform a cyrk integration using the diffeq that was written for cyrk
    time_domain_cy, y_results_cy, success_cy, message_cy = \
        cyrk_ode(diffeq_cy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_cy

    # Perform the function conversion
    diffeq_cy_converted = nb2cy(diffeq_scipy)

    # Use this function to recalculate using cyrk
    time_domain_cy_conv, y_results_cy_conv, success_cy_conv, message_cy_conv = \
        cyrk_ode(diffeq_cy_converted, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_cy_conv

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((y_results_cy_conv[0] - y_results_cy[0]) / (y_results_cy_conv[0] + y_results_cy[0]))
    p_diff_2_cy = 2. * np.abs((y_results_cy_conv[1] - y_results_cy[1]) / (y_results_cy_conv[1] + y_results_cy[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((y_results_cy_conv[0] - y_results_nb[0]) / (y_results_cy_conv[0] + y_results_nb[0]))
    p_diff_2_nb = 2. * np.abs((y_results_cy_conv[1] - y_results_nb[1]) / (y_results_cy_conv[1] + y_results_nb[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_nb2cy_args():
    """ Test converting a scipy diffeq into a cyrk one with additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    time_domain_nb, y_results_nb, success_nb, message_nb = \
        nbrk_ode(diffeq_scipy_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert success_nb

    # Perform a cyrk integration using the diffeq that was written for cyrk
    time_domain_cy, y_results_cy, success_cy, message_cy = \
        cyrk_ode(diffeq_cy_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert success_cy

    # Perform the function conversion
    diffeq_cy_converted_args = nb2cy(diffeq_scipy_args)

    # Use this function to recalculate using cyrk
    time_domain_cy_conv, y_results_cy_conv, success_cy_conv, message_cy_conv = \
        cyrk_ode(diffeq_cy_converted_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert success_cy_conv

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((y_results_cy_conv[0] - y_results_cy[0]) / (y_results_cy_conv[0] + y_results_cy[0]))
    p_diff_2_cy = 2. * np.abs((y_results_cy_conv[1] - y_results_cy[1]) / (y_results_cy_conv[1] + y_results_cy[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((y_results_cy_conv[0] - y_results_nb[0]) / (y_results_cy_conv[0] + y_results_nb[0]))
    p_diff_2_nb = 2. * np.abs((y_results_cy_conv[1] - y_results_nb[1]) / (y_results_cy_conv[1] + y_results_nb[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_cy2nb_noargs():
    """ Test converting a cyrk diffeq into a numba/scipy one with no additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    time_domain_nb, y_results_nb, success_nb, message_nb = \
        nbrk_ode(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    print(message_nb)
    assert success_nb

    # Perform a cyrk integration using the diffeq that was written for cyrk
    time_domain_cy, y_results_cy, success_cy, message_cy = \
        cyrk_ode(diffeq_cy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_cy

    # Perform the function conversion
    diffeq_nb_converted = cy2nb(diffeq_cy)

    # Use this function to recalculate using cyrk
    time_domain_nb_conv, y_results_nb_conv, success_nb_conv, message_nb_conv = \
        nbrk_ode(diffeq_nb_converted, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_nb_conv

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((y_results_nb_conv[0] - y_results_cy[0]) / (y_results_nb_conv[0] + y_results_cy[0]))
    p_diff_2_cy = 2. * np.abs((y_results_nb_conv[1] - y_results_cy[1]) / (y_results_nb_conv[1] + y_results_cy[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((y_results_nb_conv[0] - y_results_nb[0]) / (y_results_nb_conv[0] + y_results_nb[0]))
    p_diff_2_nb = 2. * np.abs((y_results_nb_conv[1] - y_results_nb[1]) / (y_results_nb_conv[1] + y_results_nb[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_cy2nb_args():
    """ Test converting a cyrk diffeq into a numba/scipy one with additional arguments. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    time_domain_nb, y_results_nb, success_nb, message_nb = \
        nbrk_ode(diffeq_scipy_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert success_nb

    # Perform a cyrk integration using the diffeq that was written for cyrk
    time_domain_cy, y_results_cy, success_cy, message_cy = \
        cyrk_ode(diffeq_cy_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert success_cy

    # Perform the function conversion
    diffeq_nb_converted_args = cy2nb(diffeq_cy_args)

    # Use this function to recalculate using cyrk
    time_domain_nb_conv, y_results_nb_conv, success_nb_conv, message_nb_conv = \
        nbrk_ode(diffeq_nb_converted_args, time_span, initial_conds, t_eval=t_eval, args=(0.01, 0.02), rtol=rtol, atol=atol)
    assert success_nb_conv

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((y_results_nb_conv[0] - y_results_cy[0]) / (y_results_nb_conv[0] + y_results_cy[0]))
    p_diff_2_cy = 2. * np.abs((y_results_nb_conv[1] - y_results_cy[1]) / (y_results_nb_conv[1] + y_results_cy[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((y_results_nb_conv[0] - y_results_nb[0]) / (y_results_nb_conv[0] + y_results_nb[0]))
    p_diff_2_nb = 2. * np.abs((y_results_nb_conv[1] - y_results_nb[1]) / (y_results_nb_conv[1] + y_results_nb[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_cy2nb_cache_njit():
    """ Test converting a cyrk diffeq into a numba/scipy with njit cacheing on. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    time_domain_nb, y_results_nb, success_nb, message_nb = \
        nbrk_ode(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_nb

    # Perform a cyrk integration using the diffeq that was written for cyrk
    time_domain_cy, y_results_cy, success_cy, message_cy = \
        cyrk_ode(diffeq_cy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_cy

    # Perform the function conversion
    diffeq_nb_converted = cy2nb(diffeq_cy)

    # Use this function to recalculate using cyrk
    time_domain_nb_conv, y_results_nb_conv, success_nb_conv, message_nb_conv = \
        nbrk_ode(diffeq_nb_converted, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_nb_conv

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((y_results_nb_conv[0] - y_results_cy[0]) / (y_results_nb_conv[0] + y_results_cy[0]))
    p_diff_2_cy = 2. * np.abs((y_results_nb_conv[1] - y_results_cy[1]) / (y_results_nb_conv[1] + y_results_cy[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((y_results_nb_conv[0] - y_results_nb[0]) / (y_results_nb_conv[0] + y_results_nb[0]))
    p_diff_2_nb = 2. * np.abs((y_results_nb_conv[1] - y_results_nb[1]) / (y_results_nb_conv[1] + y_results_nb[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)


def test_nb2cy_cache_njit():
    """ Test converting a scipy diffeq into a cyrk with njit cacheing on. """

    # Accuracy check tolerances
    check_rtol = 0.1

    # First calculate the result of an integration using nbrk.
    time_domain_nb, y_results_nb, success_nb, message_nb = \
        nbrk_ode(diffeq_scipy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_nb

    # Perform a cyrk integration using the diffeq that was written for cyrk
    time_domain_cy, y_results_cy, success_cy, message_cy = \
        cyrk_ode(diffeq_cy, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_cy

    # Perform the function conversion
    diffeq_cy_converted = nb2cy(diffeq_scipy)

    # Use this function to recalculate using cyrk
    time_domain_cy_conv, y_results_cy_conv, success_cy_conv, message_cy_conv = \
        cyrk_ode(diffeq_cy_converted, time_span, initial_conds, t_eval=t_eval, rtol=rtol, atol=atol)
    assert success_cy_conv

    # Check that the results match
    # # Converted vs. hardcoded
    p_diff_1_cy = 2. * np.abs((y_results_cy_conv[0] - y_results_cy[0]) / (y_results_cy_conv[0] + y_results_cy[0]))
    p_diff_2_cy = 2. * np.abs((y_results_cy_conv[1] - y_results_cy[1]) / (y_results_cy_conv[1] + y_results_cy[1]))
    assert np.all(p_diff_1_cy < check_rtol)
    assert np.all(p_diff_2_cy < check_rtol)

    # # Converted vs. nbrk
    p_diff_1_nb = 2. * np.abs((y_results_cy_conv[0] - y_results_nb[0]) / (y_results_cy_conv[0] + y_results_nb[0]))
    p_diff_2_nb = 2. * np.abs((y_results_cy_conv[1] - y_results_nb[1]) / (y_results_cy_conv[1] + y_results_nb[1]))
    assert np.all(p_diff_1_nb < check_rtol)
    assert np.all(p_diff_2_nb < check_rtol)
