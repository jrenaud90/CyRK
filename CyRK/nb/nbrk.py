from typing import Tuple

import numpy as np
from numba import njit

from CyRK.nb.dop_coefficients import (
    A as A_DOP, B as B_DOP, C as C_DOP, E3 as E3_DOP, E5 as E5_DOP, D as D_DOP,
    N_STAGES as N_STAGES_DOP, N_STAGES_EXTENDED as N_STAGES_EXTENDED_DOP, ORDER as ORDER_DOP,
    ERROR_ESTIMATOR_ORDER as ERROR_ESTIMATOR_ORDER_DOP)

# Optimizations
EMPTY_ARR = np.empty(0, dtype=np.float64)
EPS = np.finfo(np.float64).eps
EPS_10 = 10. * EPS
EPS_100 = 100. * EPS
INF = np.inf

# Diffeq Solver Settings
# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10.  # Maximum allowed increase in a step size.

RK23_order = 3
RK23_error_estimator_order = 2
RK23_n_stages = 3
RK23_C = np.array([0, 1 / 2, 3 / 4], order='C', dtype=np.float64)
RK23_A = np.array(
        [
            [0, 0, 0],
            [1 / 2, 0, 0],
            [0, 3 / 4, 0]
            ], order='C', dtype=np.float64
        )
RK23_B = np.array([2 / 9, 1 / 3, 4 / 9], order='C', dtype=np.float64)
RK23_E = np.array([5 / 72, -1 / 12, -1 / 9, 1 / 8], order='C', dtype=np.float64)
RK23_P = np.array(
        [[1, -4 / 3, 5 / 9],
         [0, 1, -2 / 3],
         [0, 4 / 3, -8 / 9],
         [0, -1, 1]], order='C', dtype=np.float64
        )

RK45_order = 5
RK45_error_estimator_order = 4
RK45_n_stages = 6
RK45_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1], order='C', dtype=np.float64)
RK45_A = np.array(
        [
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
            ], order='C', dtype=np.float64
        )
RK45_B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], order='C', dtype=np.float64)
RK45_E = np.array(
        [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525,
         1 / 40], order='C', dtype=np.float64
        )

RK45_P = np.array(
        [
            [1, -8048581381 / 2820520608, 8663915743 / 2820520608,
             -12715105075 / 11282082432],
            [0, 0, 0, 0],
            [0, 131558114200 / 32700410799, -68118460800 / 10900136933,
             87487479700 / 32700410799],
            [0, -1754552775 / 470086768, 14199869525 / 1410260304,
             -10690763975 / 1880347072],
            [0, 127303824393 / 49829197408, -318862633887 / 49829197408,
             701980252875 / 199316789632],
            [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
            [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]], order='C', dtype=np.float64
        )


@njit(cache=False)
def _norm(x):
    return np.linalg.norm(x) / np.sqrt(x.size)


@njit(cache=False, fastmath=False)
def nbrk_ode(
        diffeq: callable, t_span: Tuple[float, float], y0: np.ndarray, args: tuple = tuple(),
        rtol: float = 1.e-6, atol: float = 1.e-8,
        max_step: float = np.inf, first_step: float = None,
        rk_method: int = 1, t_eval: np.ndarray = EMPTY_ARR,
        capture_extra: bool = False, interpolate_extra: bool = False
        ):
    """ A Numba-safe Runge-Kutta Integrator based on Scipy's solve_ivp RK integrator.

    Parameters
    ----------
    diffeq : callable
        An njit-compiled function that defines the derivatives of the problem.
    t_span : Tuple[float, float]
        A tuple of the beginning and end of the integration domain's dependent variables.
    y0 : np.ndarray
        1D array of the initial values of the problem at t_span[0]
    args : tuple = tuple()
        Any additional arguments that are passed to dffeq.
    rtol : float = 1.e-6
        Integration relative tolerance used to determine optimal step size.
    atol : float = 1.e-8
        Integration absolute tolerance used to determine optimal step size.
    max_step : float = np.inf
        Maximum allowed step size.
    first_step : float = None
        Initial step size. If `None`, then the function will attempt to determine an appropriate initial step.
    rk_method : int = 1
        The type of RK method used for integration
            0 = RK23
            1 = RK45
            2 = DOP853
    t_eval : np.ndarray = None
        If provided, then the function will interpolate the integration results to provide them at the
            requested t-steps.
    capture_extra : bool = False
        If True, then extra output will be captured from the differential equation.
        See CyRK's Documentation/Extra Output.md for more information
    interpolate_extra : bool = False
        If True, then extra output will be interpolated (along with y) at t_eval. Otherwise, y will be interpolated
         and then differential equation will be called to find the output at each t in t_eval.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.
    .. [2] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.

    Returns
    -------
    time_domain : np.ndarray
        The final time domain. This is equal to t_eval if it was provided.
    y_results : np.ndarray
        The solution of the differential equation provided for each time_result.
            If `capture_extra` was set to True then this will output both y and any extra parameters calculated by the
             differential equation. The format of this output will look like:
            ```
            y_results[0:y_size, :]          = ... # Actual y-results calculated by the diffeq solver
            y_results[y_size:extra_size, :] = ... # Extra outputs captured alongside y during integration

    success : bool
        Final integration success flag.
    message : str
        Any integration messages, useful if success=False.

    """

    # Clean up and interpret inputs
    t_start = t_span[0]
    t_end = t_span[1]
    direction = np.sign(t_end - t_start) if t_end != t_start else 1
    direction_inf = direction * np.inf
    y0 = np.asarray(y0)
    y_size = y0.size
    y_size_sqrt = np.sqrt(y_size)
    dtype = y0.dtype
    t_eval_size = t_eval.size
    run_interpolation = t_eval_size > 0
    store_extras_during_integration = capture_extra
    if run_interpolation and not interpolate_extra:
        # If y is eventually interpolated but the extra outputs are not being interpolated, then there is
        #  no point in storing the values during the integration. Turn off this functionality to save
        #  on computation
        store_extras_during_integration = False

    # If extra output is true then the output of the diffeq will be larger than the size of y0.
    #   determine that extra size by calling the diffeq and checking its size.
    extra_start = y_size
    extra_size = 0
    total_size = y_size
    if capture_extra:
        output_ = np.asarray(diffeq(t_start, y0, *args), dtype=dtype)
        total_size = output_.size
        extra_size = total_size - y_size
        extra_start = y_size

        # Extract the extra output from the function output.
        y0_plus_extra = np.empty(total_size, dtype=dtype)
        for i in range(total_size):
            if i < extra_start:
                # Pull from y0
                y0_plus_extra[i] = y0[i]
            else:
                # Pull from extra output
                y0_plus_extra[i] = output_[i]
        if store_extras_during_integration:
            y0_to_store = y0_plus_extra
            store_loop_size = total_size
        else:
            y0_to_store = y0
            store_loop_size = y_size

    else:
        y0_to_store = y0
        store_loop_size = y_size
    extra_result = np.empty(extra_size, dtype=dtype)

    if store_extras_during_integration:
        y_result_store = np.empty(total_size, dtype=dtype)
    else:
        y_result_store = np.empty(y_size, dtype=dtype)

    # Containers to store results during integration
    time_domain_list = [t_start]
    y_result_list = [np.copy(y0_to_store)]

    # Integrator Status Codes
    #   0  = Running
    #   -1 = Failed
    #   1  = Finished with no obvious issues
    status = 0

    # Determine RK constants
    if rk_method == 0:
        # RK23 Method
        rk_order = RK23_order
        error_order = RK23_error_estimator_order
        rk_n_stages = RK23_n_stages
        rk_n_stages_plus1 = rk_n_stages + 1
        C = RK23_C
        A = RK23_A
        B = RK23_B
        E = np.asarray(RK23_E, dtype=dtype)
        # TODO: Used in dense output calculation. Not needed until that is implemented.
        # P = RK23_P
        # Set these unused variables to E to avoid variable not set check
        E3 = E
        E5 = E

        # Initialize RK-K variable
        K = np.empty((rk_n_stages_plus1, y_size), dtype=dtype)
    elif rk_method == 1:
        # RK45 Method
        rk_order = RK45_order
        error_order = RK45_error_estimator_order
        rk_n_stages = RK45_n_stages
        rk_n_stages_plus1 = rk_n_stages + 1
        C = RK45_C
        A = RK45_A
        B = RK45_B
        E = np.asarray(RK45_E, dtype=dtype)
        # TODO: Used in dense output calculation. Not needed until that is implemented.
        # P = RK45_P
        # Set these unused variables to E to avoid variable not set check
        E3 = E
        E5 = E

        # Initialize RK-K variable
        K = np.empty((rk_n_stages_plus1, y_size), dtype=dtype)
    elif rk_method == 2:
        # DOP853
        rk_order = ORDER_DOP
        error_order = ERROR_ESTIMATOR_ORDER_DOP
        rk_n_stages = N_STAGES_DOP
        rk_n_stages_plus1 = rk_n_stages + 1
        A = A_DOP[:rk_n_stages, :rk_n_stages]
        B = B_DOP
        C = C_DOP[:rk_n_stages]
        E3 = np.asarray(E3_DOP, dtype=dtype)
        E5 = np.asarray(E5_DOP, dtype=dtype)
        # TODO: Used in dense output calculation. Not needed until that is implemented.
        # D = np.asarray(D_DOP, dtype=dtype)
        # A_EXTRA = np.asarray(A_DOP[rk_n_stages + 1:], dtype=dtype)
        # C_EXTRA = np.asarray(C_DOP[rk_n_stages + 1:], dtype=dtype)
        # Set these unused variables to E to avoid variable not set check
        E = E3

        # Initialize RK-K variable
        K_extended = np.empty((N_STAGES_EXTENDED_DOP, y_size), dtype=dtype)
        K = np.ascontiguousarray(K_extended[:rk_n_stages_plus1, :])
    else:
        raise Exception(
            'Unexpected rk_method provided. Currently supported versions are:\n'
            '\t0 = RK23\n'
            '\t1 = RK34\n'
            '\t2 = DOP853')

    # Recast some constants into the correct dtype, so they can be used with y0.
    A = np.asarray(A, dtype=dtype)
    B = np.asarray(B, dtype=dtype)

    error_expo = 1. / (error_order + 1.)

    # Check tolerances
    if rtol < 100. * EPS:
        rtol = 100. * EPS

    atol = np.asarray(atol)
    if atol.ndim > 0 and atol.shape != (y_size,):
        # atol must be either the same for all y or must be provided as an array, one for each y.
        raise Exception

    # Initialize variables for start of integration
    t_old    = t_start
    t_new    = t_start
    y_new    = np.empty_like(y0)
    y_old    = np.empty_like(y0)
    dydt_old = np.empty_like(y0)
    dydt1    = np.empty_like(y0)
    dy_      = np.empty_like(y0)
    dydt_new = np.empty_like(y0)
    y_tmp    = np.empty_like(y0)
    E5_tmp   = np.empty_like(y0)
    E3_tmp   = np.empty_like(y0)
    E_tmp    = np.empty_like(y0)
    for i in range(y_size):
        y0_i     = y0[i]
        y_new[i] = y0_i
        y_old[i] = y0_i

    output = np.asarray(diffeq(t_new, y_new, *args), dtype=dtype)
    for i in range(y_size):
        dydt_old[i] = output[i]

    # Find first step size
    first_step_found = False
    if first_step is not None:
        step_size = max_step
        if first_step < 0.:
            # Step size must be a positive number
            raise Exception
        elif first_step > np.abs(t_end - t_start):
            # Step size can not exceed bounds
            raise Exception
        elif first_step != 0.:
            step_size = first_step
            first_step_found = True

    if not first_step_found:
        # Select an initial step size based on the differential equation.
        # .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
        #        Equations I: Nonstiff Problems", Sec. II.4.
        if y_size == 0:
            step_size = INF
        else:

            # Take the norm of d0 and d1
            d0 = 0.
            d1 = 0.
            for i in range(y_size):
                scale = atol + np.abs(y_old[i]) * rtol

                d0_abs = np.abs(y_old[i] / scale)
                d1_abs = np.abs(dydt_old[i] / scale)
                d0 += (d0_abs * d0_abs)
                d1 += (d1_abs * d1_abs)

            d0 = np.sqrt(d0) / y_size_sqrt
            d1 = np.sqrt(d1) / y_size_sqrt

            if d0 < 1.e-5 or d1 < 1.e-5:
                h0 = 1.e-6
            else:
                h0 = 0.01 * d0 / d1

            y1 = y_old + h0 * direction * dydt_old
            t1 = t_old + h0 * direction

            # Use the differential equation to estimate the first step size
            diffeq_output = np.asarray(diffeq(t1, y1, *args), dtype=dtype)
            for i in range(y_size):
                dydt1[i] = diffeq_output[i]

            d2 = 0.
            for i in range(y_size):
                scale = atol + np.abs(y_old[i]) * rtol
                d2_abs = np.abs((dydt1[i] - dydt_old[i]) / scale)
                d2 += (d2_abs * d2_abs)
            d2 = np.sqrt(d2) / (h0 * y_size_sqrt)

            if d1 <= 1.e-15 and d2 <= 1.e-15:
                h1 = max(1.e-6, h0 * 1.e-3)
            else:
                h1 = (0.01 / max(d1, d2))**error_expo

            next_after = 10. * abs(np.nextafter(t_old, direction * np.inf) - t_old)
            step_size  = max(next_after, min(100. * h0, h1))

    # Main integration loop
    # # Time Loop
    while status == 0:

        if t_new == t_end or y_size == 0:
            t_old = t_end
            t_new = t_end
            status = 1
            break

        # Run RK integration step
        # Determine step size based on previous loop
        # Find minimum step size based on the value of t (less floating point numbers between numbers when t is large)
        next_after = 10. * abs(np.nextafter(t_old, direction * np.inf) - t_old)
        min_step   = next_after

        # Look for over/undershoots in previous step size
        if step_size > max_step:
            step_size = max_step
        elif step_size < min_step:
            step_size = min_step

        # Determine new step size
        step_accepted = False
        step_rejected = False
        step_error = False
        # # Step Loop
        while not step_accepted:

            if step_size < min_step:
                step_error = True
                break

            # Move time forward for this particular step size
            step = step_size * direction
            t_new = t_old + step

            # Check that we are not at the end of integration with that move
            if direction * (t_new - t_end) > 0:
                t_new = t_end

                # Correct the step if we were at the end of integration
                step = t_new - t_old
                step_size = np.abs(step)

            # Calculate derivative using RK method
            K[0, :] = dydt_old[:]
            for s in range(1, len(C)):
                c = C[s]
                A_ = A[s, :]
                time_ = t_old + c * step

                # Dot Product (K, a) * step
                for j in range(s):
                    K_ = K[j, :]
                    a_ = A_[j]
                    for i in range(y_size):
                        if j == 0:
                            # Initialize
                            y_tmp[i] = y_old[i]

                        y_tmp[i] = y_tmp[i] + (K_[i] * a_ * step)

                # Update K with a new result from the differential equationC
                diffeq_output = np.asarray(diffeq(time_, y_tmp, *args), dtype=dtype)
                for i in range(y_size):
                    dy_[i] = diffeq_output[i]

                for i in range(y_size):
                    K[s, i] = dy_[i]

            # Dot Product (K, B) * step
            for j in range(rk_n_stages):
                # We do not use rk_n_stages_plus1 here because we are chopping off the last row of K to match
                #  the shape of B.
                K_ = K[j, :]
                b_ = B[j]
                for i in range(y_size):
                    if j == 0:
                        # Initialize
                        y_new[i] = y_old[i]
                    y_new[i] = y_new[i] + (K_[i] * b_ * step)

            # Find final dydt for this timestep
            diffeq_output = np.asarray(diffeq(t_new, y_new, *args), dtype=dtype)
            for i in range(store_loop_size):
                if i < extra_start:
                    # Set diffeq results
                    dydt_new[i] = diffeq_output[i]
                else:
                    # Set extra results
                    extra_result[i - extra_start] = diffeq_output[i]

            # Estimate error to change the step size for next step
            if rk_method == 2:
                # DOP853 error estimation
                # Dot Product (K, E5) / Scale and (K, E3) / scale
                for i in range(y_size):
                    # Check how well this step performed
                    scale = atol + np.maximum(np.abs(y_old[i]), np.abs(y_new[i])) * rtol
                    for j in range(rk_n_stages_plus1):
                        if j == 0:
                            # Initialize
                            E5_tmp[i] = 0.
                            E3_tmp[i] = 0.
                        elif j == rk_n_stages:
                            # Set last array of the K array
                            K[j, i] = dydt_new[i]
                        K_scale = K[j, i] / scale
                        E5_tmp[i] = E5_tmp[i] + (K_scale * E5[j])
                        E3_tmp[i] = E3_tmp[i] + (K_scale * E3[j])

                # Find norms for each error
                error_norm5 = 0.
                error_norm3 = 0.

                # Perform summation
                for i in range(y_size):
                    error_norm5_abs = np.abs(E5_tmp[i])
                    error_norm3_abs = np.abs(E3_tmp[i])
                    error_norm5 += (error_norm5_abs * error_norm5_abs)
                    error_norm3 += (error_norm3_abs * error_norm3_abs)

                # Check if errors are zero
                if (error_norm5 == 0.) and (error_norm3 == 0.):
                    error_norm = 0.
                else:
                    error_denom = error_norm5 + 0.01 * error_norm3
                    error_norm = step_size * error_norm5 / np.sqrt(error_denom * y_size)

            else:
                # Calculate Error for RK23 and RK45
                error_norm = 0.
                # Dot Product (K, E) * step / scale
                for i in range(y_size):
                    # Check how well this step performed.
                    scale = atol + max(np.abs(y_old[i]), np.abs(y_new[i])) * rtol
                    for j in range(rk_n_stages_plus1):
                        if j == 0:
                            # Initialize
                            E_tmp[i] = 0.
                        elif j == rk_n_stages:
                            # Set last array of the K array.
                            K[j, i] = dydt_new[i]
                        K_scale = K[j, i] / scale
                        E_tmp[i] = E_tmp[i] + (K_scale * E[j] * step)

                    error_norm_abs = np.abs(E_tmp[i])
                    error_norm += (error_norm_abs * error_norm_abs)
                error_norm = np.sqrt(error_norm) / y_size_sqrt

            if error_norm < 1.:
                # The error is low! Let's update this step for the next time loop
                if error_norm == 0.:
                    step_factor = MAX_FACTOR
                else:
                    step_factor = min(
                            MAX_FACTOR,
                            SAFETY * error_norm**-error_expo
                            )

                if step_rejected:
                    # There were problems with this step size on the previous step loop. Make sure factor does
                    #    not exasperate them.
                    step_factor = min(step_factor, 1.)

                step_size = step_size * step_factor
                step_accepted = True
            else:
                step_size = step_size * max(
                        MIN_FACTOR,
                        SAFETY * error_norm**-error_expo
                        )
                step_rejected = True

        if not step_accepted or step_error:
            # Issue with step convergence
            status = -1
            break

        # End of step loop. Update the _now variables
        t_old = t_new

        for i in range(y_size):
            y_old[i] = y_new[i]
            dydt_old[i] = dydt_new[i]

        # Save data
        # If there is extra outputs then we need to store those at this timestep as well.
        for i in range(store_loop_size):
            if i < extra_start:
                # Pull from y result
                y_result_store[i] = y_new[i]
            else:
                # Pull from extra
                y_result_store[i] = extra_result[i - extra_start]

        time_domain_list.append(t_new)
        y_result_list.append(np.copy(y_result_store))

    t_size = len(time_domain_list)

    # To match the format that scipy follows, we will take the transpose of y.
    time_domain = np.empty(t_size, dtype=np.float64)
    y_results = np.empty((store_loop_size, t_size), dtype=dtype)
    for t_i in range(t_size):
        time_domain[t_i] = time_domain_list[t_i]
        y_results_list_at_t = y_result_list[t_i]
        for y_i in range(store_loop_size):
            y_results[y_i, t_i] = y_results_list_at_t[y_i]

    if t_eval_size > 0:
        # User only wants data at specific points.
        # The current version of this function has not implemented sicpy's dense output, so we must use an interpolation.
        t_eval = np.asarray(t_eval, dtype=np.float64)
        y_results_reduced = np.empty((total_size, t_eval.size), dtype=dtype)

        # np.interp only works on 1D arrays, so we have to loop through each of the variables and call for each y:
        for i in range(y_size):
            y_results_reduced[i, :] = np.interp(t_eval, time_domain, y_results[i, :])

        if capture_extra:
            # Right now if there is any extra output then it is stored at each time step used in the RK loop.
            # We have to make a choice on what to output do we, like we do with y, interpolate all of those extras?
            #  or do we use the interpolation on y to find new values.
            # The latter method is more computationally expensive (recalls the diffeq for each y) but is more accurate.
            if interpolate_extra:
                # Continue the interpolation for the extra values.
                for i in range(extra_size):
                    y_results_reduced[extra_start + i, :] = \
                        np.interp(t_eval, time_domain, y_results[extra_start + i, :])
            else:
                # Use y and t to recalculate the extra outputs
                y_ = np.empty(y_size, dtype=dtype)
                for t_i in range(t_eval_size):
                    t_ = t_eval[t_i]
                    for y_i in range(y_size):
                        y_[y_i] = y_results_reduced[y_i, t_i]
                    diffeq_output = np.asarray(diffeq(t_, y_, *args), dtype=dtype)
                    for i in range(extra_size):
                        y_results_reduced[extra_start + i, t_i] = diffeq_output[extra_start + i]

        y_results = y_results_reduced
        time_domain = t_eval
    success = status == 1

    if status == 1:
        message = 'Integration finished.'
    elif status == 0:
        message = 'Integration interrupted.'
    elif status == -1:
        message = 'Error in step size calculation:\n\tRequired step size is less than spacing between numbers.'
    else:
        message = 'An unknown integration error occurred.'

    return time_domain, y_results, success, message
