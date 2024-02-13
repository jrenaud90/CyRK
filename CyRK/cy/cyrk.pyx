# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import cython

from cpython.mem cimport PyMem_Free

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, fabs, nextafter, NAN, floor

from CyRK.utils.utils cimport allocate_mem, reallocate_mem
from CyRK.rk.rk cimport find_rk_properties
from CyRK.cy.common cimport double_numeric, interpolate, SAFETY, MIN_FACTOR, MAX_FACTOR, MAX_STEP, INF, \
    EPS_100, find_expected_size, find_max_num_steps


cdef double cabs(
        double complex value
        ) noexcept nogil:
    """ Absolute value function for complex-valued inputs.

    Parameters
    ----------
    value : float (double complex)
        Complex-valued number.

    Returns
    -------
    value_abs : float (double)
        Absolute value of `value`.
    """

    cdef double v_real
    cdef double v_imag
    v_real = value.real
    v_imag = value.imag

    return sqrt(v_real * v_real + v_imag * v_imag)


cdef double dabs(
        double_numeric value
        ) noexcept nogil:
    """ Absolute value function for either float or complex-valued inputs.

    Checks the type of value and either utilizes `cabs` (for double complex) or `fabs` (for floats).

    Parameters
    ----------
    value : float (double_numeric)
        Float or complex-valued number.

    Returns
    -------
    value_abs : float (double)
        Absolute value of `value`.
    """

    # Check the type of value
    if double_numeric is cython.doublecomplex:
        return cabs(value)
    else:
        return fabs(value)


def cyrk_ode(
        diffeq,
        (double, double) t_span,
        const double_numeric[:] y0,
        tuple args = None,
        double rtol = 1.e-3,
        double atol = 1.e-6,
        double[::1] rtols = None,
        double[::1] atols = None,
        double max_step = MAX_STEP,
        double first_step = 0.,
        unsigned char rk_method = 1,
        double[:] t_eval = None,
        bint capture_extra = False,
        size_t num_extra = 0,
        bint interpolate_extra = False,
        size_t expected_size = 0,
        size_t max_num_steps = 0,
        size_t max_ram_MB = 2000
        ):
    """
    cyrk_ode: A Runge-Kutta Solver Implemented in Cython.

    Parameters
    ----------
    diffeq : callable
        A python or njit-ed numba differential equation.
        Format should follow:
        ```
        def diffeq(t, y, dy, arg_1, arg_2, ...):
            dy[0] = y[0] * t
            ....
        ```
    t_span : (double, double)
        Values of independent variable at beginning and end of integration.
    y0 : double[::1]
        Initial values for the dependent y variables at `t_span[0]`.
    args : tuple or None, default=None
        Additional arguments used by the differential equation.
        None (default) will tell the solver to not use additional arguments.
    rk_method : int, default=1
        Runge-Kutta method that will be used. Currently implemented models:
            0: ‘RK23’: Explicit Runge-Kutta method of order 3(2).
            1: ‘RK45’ (default): Explicit Runge-Kutta method of order 5(4).
            2: ‘DOP853’: Explicit Runge-Kutta method of order 8.
    rtol : double, default=1.0e-3
        Relative tolerance using in local error calculation.
    atol : double, default=1.0e-6
        Absolute tolerance using in local error calculation.
    rtols : double[::1], default=None
        np.ndarray of relative tolerances, one for each dependent y variable.
        None (default) will use the same tolerance (set by `rtol`) for each y variable.
    atols : double[::1], default=None
        np.ndarray of absolute tolerances, one for each dependent y variable.
        None (default) will use the same tolerance (set by `atol`) for each y variable.
    max_step : double, default=+Inf
        Maximum allowed step size.
    first_step : double, default=0
        First step's size (after `t_span[0]`).
        If set to 0 (the default) then the solver will attempt to guess a suitable initial step size.
    max_num_steps : size_t, default=0
        Maximum number of step sizes allowed before solver will auto fail.
        If set to 0 (the default) then the maximum number of steps will be equal to max integer size
        allowed on system architecture.
    t_eval : double[::1], default=None
        If not set to None, then a final interpolation will be performed on the solution to fit it to this array.
    capture_extra : bool = False
        If True, then additional output from the differential equation will be collected (but not used to determine
         integration error).
         Example:
            ```
            def diffeq(t, y, dy):
                a = ... some function of y and t.
                dy[0] = a**2 * sin(t) - y[1]
                dy[1] = a**3 * cos(t) + y[0]

                # Storing extra output in dy even though it is not part of the diffeq.
                dy[2] = a
            ```
    num_extra : int = 0
        The number of extra outputs the integrator should expect. With the previous example there is 1 extra output.
    interpolate_extra : bint, default=False
        Flag if interpolation should be run on extra parameters.
        If set to False when `run_interpolation=True`, then interpolation will be run on solution's y, t. These will
        then be used to recalculate extra parameters rather than an interpolation on the extra parameters captured
        during integration.
    expected_size : size_t, default=0
        Anticipated size of integration range, i.e., how many steps will be required.
        Used to build temporary storage arrays for the solution results.
        If set to 0 (the default), then the solver will attempt to guess on a suitable expected size based on the
        relative tolerances and size of the integration domain.

    Returns
    -------
    time_domain : np.ndarray
        The final time domain. This is equal to t_eval if it was provided.
    y_results : np.ndarray
        The solution of the differential equation provided for each time_result.
    success : bool
        Final integration success flag.
    message : str
        Any integration messages, useful if success=False.
    """

    # Setup loop variables
    cdef size_t s, i, j

    # Setup integration variables
    cdef char status
    cdef str message

    # Determine information about the differential equation based on its initial conditions
    cdef size_t y_size
    cdef double y_size_dbl, y_size_sqrt
    cdef bint y_is_complex
    y_size       = y0.size
    y_is_complex = False
    y_size_dbl   = <double>y_size
    y_size_sqrt  = sqrt(y_size_dbl)

    # Check the type of the values in y0
    if double_numeric is cython.double:
        DTYPE = np.float64
    elif double_numeric is cython.doublecomplex:
        DTYPE = np.complex128
        y_is_complex = True
    else:
        # Cyrk only supports float64 and complex128.
        status = -8
        message = "Attribute error."
        raise Exception('Unexpected type found for initial conditions (y0).')

    # Build time domain
    cdef double t_start, t_end, t_delta, t_delta_check, t_delta_abs, direction_inf, t_old, t_now, time_
    cdef bint direction_flag
    t_start       = t_span[0]
    t_end         = t_span[1]
    t_delta       = t_end - t_start
    t_delta_abs   = fabs(t_delta)
    t_delta_check = t_delta_abs

    if t_delta >= 0.:
        # Integration is moving forward in time.
        direction_flag = True
        direction_inf = INF
    else:
        # Integration is moving backwards in time.
        direction_flag = False
        direction_inf = -INF

    # Pull out information on args
    cdef bint use_args
    if args is None:
        use_args = False
    else:
        use_args = True

    # Setup temporary variables to store intermediate values
    cdef double temp_double
    cdef double_numeric temp_double_numeric

    # Determine integration tolerances
    cdef double* tol_ptrs = NULL
    cdef double* rtols_ptr = NULL
    cdef double* atols_ptr = NULL
    tol_ptrs  = <double *> allocate_mem(2 * y_size * sizeof(double), 'tol_ptrs (start-up)')
    rtols_ptr = &tol_ptrs[0]
    atols_ptr = &tol_ptrs[y_size]
    
    cdef double rtol_min
    rtol_min = INF
    if rtols is not None:
        # User provided an arrayed version of rtol.
        if len(rtols) != y_size:
            raise AttributeError('rtol array must be the same size as y0.')
        for i in range(y_size):
            temp_double = rtols[i]
            # Check that the tolerances are not too small.
            if temp_double < EPS_100:
                temp_double = EPS_100
            rtol_min = min(rtol_min, temp_double)
            rtols_ptr[i] = temp_double
    else:
        # No array provided. Use the same rtol for all ys.
        # Check that the tolerances are not too small.
        if rtol < EPS_100:
            rtol = EPS_100
        rtol_min = rtol
        for i in range(y_size):
            rtols_ptr[i] = rtol

    if atols is not None:
        # User provided an arrayed version of atol.
        if len(atols) != y_size:
            raise AttributeError('atol array must be the same size as y0.')
        for i in range(y_size):
            atols_ptr[i] = atols[i]
    else:
        # No array provided. Use the same atol for all ys.
        for i in range(y_size):
            atols_ptr[i] = atol
    
    # Determine max number of steps
    cdef size_t max_num_steps_touse
    cdef bint user_provided_max_num_steps
    find_max_num_steps(
        y_size,
        num_extra,
        max_num_steps,
        max_ram_MB,
        capture_extra,
        y_is_complex,
        &user_provided_max_num_steps,
        &max_num_steps_touse)

    # Expected size of output arrays.
    cdef size_t expected_size_to_use, num_concats, current_size
    if expected_size == 0:
        # cyrk_ode will attempt to guess on a best size for the arrays.
        expected_size_to_use = find_expected_size(
                y_size,
                num_extra,
                t_delta_abs,
                rtol_min,
                capture_extra,
                y_is_complex)
    else:
        expected_size_to_use = expected_size
    # Set the current size to the expected size.
    # `expected_size` should never change but current might grow if expected size is not large enough.
    current_size = expected_size_to_use
    num_concats  = 1

    # Initialize live variable arrays
    cdef double_numeric* y_storage_ptrs = NULL
    cdef double_numeric* y_old_ptr = NULL
    cdef double_numeric* dy_ptr = NULL
    cdef double_numeric* dy_old_ptr = NULL

    y_storage_ptrs = <double_numeric *> allocate_mem(3 * y_size * sizeof(double_numeric), 'y_storage_ptrs (start-up)')

    y_old_ptr  = &y_storage_ptrs[0]
    dy_ptr     = &y_storage_ptrs[1 * y_size]
    dy_old_ptr = &y_storage_ptrs[2 * y_size]

    # Build memoryviews based on y_view and dy_ptr that can be passed to the diffeq.
    # This is process is different than CySolver which strictly uses c pointers.
    # These memoryviews allow for user-provided diffeqs that are not cython/compiled.
    y_array = np.empty(y_size, dtype=DTYPE, order='C')
    cdef double_numeric[::1] y_view = y_array

    # Store y0 into the y arrays
    for i in range(y_size):
        temp_double_numeric = y0[i]
        y_view[i]    = temp_double_numeric
        y_old_ptr[i] = temp_double_numeric

    # Determine extra outputs
    # To avoid memory access violations we need to set the extra output arrays no matter if they are used.
    # If not used, just set them to size zero.
    if capture_extra:
        if num_extra <= 0:
            status = -8
            raise AttributeError('Capture extra set to True, but number of extra set to 0 (or negative).')
    else:
        # Even though we are not capturing extra, we still want num_extra to be equal to 1 so that nan arrays
        # are properly initialized
        num_extra = 1

    cdef double_numeric* extra_output_init_ptr = NULL
    cdef double_numeric* extra_output_ptr = NULL
    extra_output_init_ptr = <double_numeric *> allocate_mem(
        num_extra * sizeof(double_numeric),
        'extra_output_init_ptr (start-up)')
    extra_output_ptr = <double_numeric *> allocate_mem(
        num_extra * sizeof(double_numeric),
        'extra_output_ptr (start-up)')

    for i in range(num_extra):
        extra_output_init_ptr[i] = NAN
        extra_output_ptr[i]      = NAN

    # If extra output is true then the output of the diffeq will be larger than the size of y0.
    # Determine that extra size by calling the diffeq and checking its size.
    cdef size_t extra_start, total_size
    extra_start = y_size
    if capture_extra:
        total_size = y_size + num_extra
    else:
        total_size = y_size

    # Build pointer to store results of diffeq
    diffeq_out_array = np.empty(total_size, dtype=DTYPE, order='C')
    cdef double_numeric[::1] diffeq_out_view = diffeq_out_array

    # Determine interpolation information
    cdef bint run_interpolation
    cdef size_t len_t_eval
    if t_eval is None:
        run_interpolation = False
        interpolate_extra = False
        # Even though we are not using t_eval, set its size equal to one so that nan arrays can be built
        len_t_eval = 1
    else:
        run_interpolation = True
        interpolate_extra = interpolate_extra
        len_t_eval = len(t_eval)

    cdef double* t_eval_ptr = NULL
    t_eval_ptr = <double *> allocate_mem(len_t_eval * sizeof(double), 't_eval_ptr (start-up)')
    for i in range(len_t_eval):
        if run_interpolation:
            t_eval_ptr[i] = t_eval[i]
        else:
            t_eval_ptr[i] = NAN

    # Make initial call to diffeq to get initial dydt and any extra outputs (if requested) at t0.
    if use_args:
        diffeq(t_start, y_array, diffeq_out_array, *args)
    else:
        diffeq(t_start, y_array, diffeq_out_array)

    # Setup initial conditions
    t_old = t_start
    t_now = t_start
    for i in range(y_size):
        temp_double_numeric = diffeq_out_view[i]
        dy_ptr[i]     = temp_double_numeric
        dy_old_ptr[i] = temp_double_numeric

    # Capture the extra output for the initial condition.
    if capture_extra:
        for i in range(num_extra):
            # Pull from extra output
            extra_output_init_ptr[i] = diffeq_out_view[extra_start + i]

    # Determine RK scheme and initialize RK memory views
    cdef double* A_ptr = NULL
    cdef double* B_ptr = NULL
    cdef double* C_ptr = NULL
    cdef double* E_ptr = NULL
    cdef double* E3_ptr = NULL
    cdef double* E5_ptr = NULL
    cdef size_t rk_order, error_order, rk_n_stages, len_Arows, len_Acols, len_C, rk_n_stages_plus1
    cdef double error_expo, error_pow

    find_rk_properties(
        rk_method,
        &rk_order,
        &error_order,
        &rk_n_stages,
        &len_Arows,
        &len_Acols,
        &A_ptr,
        &B_ptr,
        &C_ptr,
        &E_ptr,
        &E3_ptr,
        &E5_ptr
        )

    if rk_order == 0:
        raise AttributeError('Unknown or not-yet-implemented RK method requested.')
    
    len_C             = rk_n_stages
    rk_n_stages_plus1 = rk_n_stages + 1
    error_expo        = 1. / (<double>error_order + 1.)
    
    # Initialize other RK-related Arrays
    cdef double_numeric* K_ptr = NULL
    K_ptr = <double_numeric *> allocate_mem(rk_n_stages_plus1 * y_size * sizeof(double_numeric), 'K_ptr (start-up)')
    # It is important K be initialized with 0s
    for i in range(rk_n_stages_plus1):
        for j in range(y_size):
            K_ptr[i * y_size + j] = 0.
    
    cdef double error_norm5, error_norm3, error_norm, error_norm_abs, error_norm3_abs, error_norm5_abs, error_denom

    # Other RK Optimizations
    cdef double A_at_sj, A_at_10, B_at_j
    cdef double_numeric K_
    A_at_10 = A_ptr[1 * len_Acols + 0]

    # Setup storage arrays
    # These arrays are built to fit a number of points equal to current_size
    # If the integration needs more than that then a new array will be concatenated (with performance costs) to these.
    cdef double* time_domain_array_ptr = NULL
    cdef double_numeric* y_results_array_ptr = NULL
    cdef double_numeric* extra_array_ptr = NULL

    time_domain_array_ptr = <double *> allocate_mem(
        current_size * sizeof(double),
        'time_domain_array_ptr (start-up)')
    y_results_array_ptr = <double_numeric *> allocate_mem(
        y_size * current_size * sizeof(double_numeric),
        'y_results_array_ptr (start-up)')
    if capture_extra:
        extra_array_ptr = <double_numeric *> allocate_mem(
            num_extra * current_size * sizeof(double_numeric),
            'extra_array_ptr (start-up)')

    # Load initial conditions into storage arrays
    time_domain_array_ptr[0] = t_start
    for i in range(y_size):
        y_results_array_ptr[i] = y0[i]
    if capture_extra:
        for i in range(num_extra):
            extra_array_ptr[i] = extra_output_init_ptr[i]

    # Solution pointers
    cdef double* solution_t_ptr = NULL
    cdef double_numeric* solution_y_ptr = NULL
    cdef double_numeric* solution_extra_ptr = NULL

    # Determine size of first step.
    cdef double d0, d1, d2, d0_abs, d1_abs, d2_abs, h0, h1, scale
    cdef double step, step_size, min_step, step_factor

    # Integration flags and variables
    cdef bint success, step_accepted, step_rejected, step_error
    cdef size_t len_t

    # Integration completion variables
    cdef size_t len_t_touse
    cdef double* interpolated_solution_t_ptr = NULL
    cdef double_numeric* interpolated_solution_y_ptr = NULL
    cdef double_numeric* interpolated_solution_extra_ptr = NULL
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] solution_t
    cdef np.ndarray[double_numeric, ndim=2, mode='c'] solution_y
    cdef double[::1] solution_t_view
    cdef double_numeric[:, ::1] solution_y_view

    try:
        if first_step == 0.:
            # Select an initial step size based on the differential equation.
            # .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            #        Equations I: Nonstiff Problems", Sec. II.4.
            if y_size == 0:
                step_size = INF
            else:
                # Find the norm for d0 and d1
                d0 = 0.
                d1 = 0.
                for i in range(y_size):

                    temp_double = dabs(y_old_ptr[i])
                    scale = atols_ptr[i] + dabs(temp_double) * rtols_ptr[i]
                    d0_abs = dabs(temp_double) / scale
                    d1_abs = dabs(dy_old_ptr[i]) / scale
                    d0 += (d0_abs * d0_abs)
                    d1 += (d1_abs * d1_abs)

                d0 = sqrt(d0) / y_size_sqrt
                d1 = sqrt(d1) / y_size_sqrt

                if d0 < 1.e-5 or d1 < 1.e-5:
                    h0 = 1.e-6
                else:
                    h0 = 0.01 * d0 / d1

                if direction_flag:
                    h0_direction = h0
                else:
                    h0_direction = -h0
                t_now = t_old + h0_direction
                for i in range(y_size):
                    y_view[i] = y_old_ptr[i] + h0_direction * dy_old_ptr[i]

                if use_args:
                    diffeq(t_now, y_array, diffeq_out_array, *args)
                else:
                    diffeq(t_now, y_array, diffeq_out_array)

                # Find the norm for d2
                d2 = 0.
                for i in range(y_size):
                    temp_double_numeric = diffeq_out_view[i]
                    dy_ptr[i] = temp_double_numeric
                    scale = atols_ptr[i] + dabs(y_old_ptr[i]) * rtols_ptr[i]
                    d2_abs = dabs( (temp_double_numeric - dy_old_ptr[i]) ) / scale
                    d2 += (d2_abs * d2_abs)

                d2 = sqrt(d2) / (h0 * y_size_sqrt)

                if d1 <= 1.e-15 and d2 <= 1.e-15:
                    h1 = max(1.e-6, h0 * 1.e-3)
                else:
                    h1 = (0.01 / max(d1, d2))**error_expo

                step_size = max(10. * fabs(nextafter(t_old, direction_inf) - t_old), min(100. * h0, h1))
        else:
            if first_step <= 0.:
                status = -8
                message = "Attribute error."
                raise AttributeError('Error in user-provided step size: Step size must be a positive number.')
            elif first_step > t_delta_abs:
                status = -8
                message = "Attribute error."
                raise AttributeError('Error in user-provided step size: Step size can not exceed bounds.')
            step_size = first_step

        # # Main integration loop
        # Set integration flags
        success       = False
        step_accepted = False
        step_rejected = False
        step_error    = False
        status        = 0
        message       = "Integration is/was ongoing (perhaps it was interrupted?)."

        # Track number of steps.
        # Initial conditions were provided so the number of steps is already 1
        len_t = 1

        if y_size == 0:
            status = -6
            message = "Integration never started: y-size is zero."

        while status == 0:
            if t_now == t_end:
                t_old = t_end
                status = 1
                break

            if len_t > max_num_steps_touse:
                if user_provided_max_num_steps:
                    status = -2
                    message = "Maximum number of steps (set by user) exceeded during integration."
                else:
                    status = -3
                    message = "Maximum number of steps (set by ram usage limit) exceeded during integration."
                break

            # Run RK integration step
            # Determine step size based on previous loop
            # Find minimum step size based on the value of t (less floating point numbers between numbers when t is large)
            min_step = 10. * fabs(nextafter(t_old, direction_inf) - t_old)
            # Look for over/undershoots in previous step size
            if step_size > max_step:
                step_size = max_step
            elif step_size < min_step:
                step_size = min_step

            # Determine new step size
            step_accepted = False
            step_rejected = False
            step_error    = False

            # # Step Loop
            while not step_accepted:

                if step_size < min_step:
                    step_error = True
                    status     = -1
                    break

                # Move time forward for this particular step size
                if direction_flag:
                    step = step_size
                    t_now = t_old + step
                    t_delta_check = t_now - t_end
                else:
                    step = -step_size
                    t_now = t_old + step
                    t_delta_check = t_end - t_now

                # Check that we are not at the end of integration with that move
                if t_delta_check > 0.:
                    t_now = t_end

                    # Correct the step if we were at the end of integration
                    step = t_now - t_old
                    if direction_flag:
                        step_size = step
                    else:
                        step_size = -step

                # Calculate derivative using RK method
                # Dot Product (K, a) * step
                for s in range(1, len_C):
                    time_ = t_old + C_ptr[s] * step

                    # Dot Product (K, a) * step
                    if s == 1:
                        for i in range(y_size):
                            # Set the first column of K
                            temp_double_numeric = dy_old_ptr[i]
                            K_ptr[i] = temp_double_numeric

                            # Calculate y_new for s==1
                            y_view[i] = y_old_ptr[i] + (temp_double_numeric * A_at_10 * step)
                    else:
                        for j in range(s):
                            A_at_sj = A_ptr[s * len_Acols + j] * step
                            for i in range(y_size):
                                if j == 0:
                                    # Initialize
                                    y_view[i] = y_old_ptr[i]

                                y_view[i] = y_view[i] + K_ptr[j * y_size + i] * A_at_sj

                    if use_args:
                        diffeq(time_, y_array, diffeq_out_array, *args)
                    else:
                        diffeq(time_, y_array, diffeq_out_array)

                    for i in range(y_size):
                        K_ptr[s * y_size + i] = diffeq_out_view[i]

                # Dot Product (K, B) * step
                for j in range(rk_n_stages):
                    B_at_j = B_ptr[j] * step
                    # We do not use rk_n_stages_plus1 here because we are chopping off the last row of K to match
                    #  the shape of B.
                    for i in range(y_size):
                        if j == 0:
                            # Initialize
                            y_view[i] = y_old_ptr[i]

                        y_view[i] = y_view[i] + K_ptr[j * y_size + i] * B_at_j

                # Find final dydt for this timestep
                if use_args:
                    diffeq(t_now, y_array, diffeq_out_array, *args)
                else:
                    diffeq(t_now, y_array, diffeq_out_array)

                # Store extra
                if capture_extra:
                    for i in range(num_extra):
                        extra_output_ptr[i] = diffeq_out_view[extra_start + i]

                if rk_method == 2:
                    # Calculate Error for DOP853
                    # Find norms for each error
                    error_norm5 = 0.
                    error_norm3 = 0.
                    # Dot Product (K, E5) / scale and Dot Product (K, E3) * step / scale
                    for i in range(y_size):
                        # Find scale of y for error calculations
                        scale = atols_ptr[i] + max(dabs(y_old_ptr[i]), dabs(y_view[i])) * rtols_ptr[i]

                        # Set diffeq results
                        temp_double_numeric = diffeq_out_view[i]
                        dy_ptr[i] = temp_double_numeric

                        # Set last array of K equal to dydt
                        K_ptr[rk_n_stages * y_size + i] = temp_double_numeric
                        # Initialize
                        error_dot_1 = 0.
                        error_dot_2 = 0.
                        for j in range(rk_n_stages_plus1):

                            K_ = K_ptr[j * y_size + i]
                            error_dot_1 += K_ * E3_ptr[j]
                            error_dot_2 += K_ * E5_ptr[j]

                        error_norm3_abs = dabs(error_dot_1) / scale
                        error_norm5_abs = dabs(error_dot_2) / scale

                        error_norm3 += (error_norm3_abs * error_norm3_abs)
                        error_norm5 += (error_norm5_abs * error_norm5_abs)

                    # Check if errors are zero
                    if (error_norm5 == 0.) and (error_norm3 == 0.):
                        error_norm = 0.
                    else:
                        error_denom = error_norm5 + 0.01 * error_norm3
                        error_norm  = step_size * error_norm5 / sqrt(error_denom * y_size_dbl)

                else:
                    # Calculate Error for RK23 and RK45
                    # Dot Product (K, E) * step / scale
                    error_norm = 0.
                    for i in range(y_size):
                        # Find scale of y for error calculations
                        scale = atols_ptr[i] + max(dabs(y_old_ptr[i]), dabs(y_view[i])) * rtols_ptr[i]

                        # Set diffeq results
                        temp_double_numeric = diffeq_out_view[i]
                        dy_ptr[i] = temp_double_numeric

                        # Set last array of K equal to dydt
                        K_ptr[rk_n_stages * y_size + i] = temp_double_numeric
                        # Initialize
                        error_dot_1 = 0.
                        for j in range(rk_n_stages_plus1):

                            error_dot_1 += K_ptr[j * y_size + i] * E_ptr[j]

                        error_norm_abs = dabs(error_dot_1) * (step / scale)
                        error_norm    += (error_norm_abs * error_norm_abs)
                    error_norm = sqrt(error_norm) / y_size_sqrt

                if error_norm < 1.:
                    # The error is low! Let's update this step for the next time loop
                    if error_norm == 0.:
                        step_factor = MAX_FACTOR
                    else:
                        error_pow = error_norm**-error_expo
                        step_factor = min(MAX_FACTOR, SAFETY * error_pow)

                    if step_rejected:
                        # There were problems with this step size on the previous step loop. Make sure factor does
                        #    not exasperate them.
                        step_factor = min(step_factor, 1.)

                    step_size = step_size * step_factor
                    step_accepted = True
                else:
                    error_pow = error_norm**-error_expo
                    step_size = step_size * max(MIN_FACTOR, SAFETY * error_pow)
                    step_rejected = True

            if step_error:
                # Issue with step convergence
                status = -1
                message = "Error in step size calculation:\n\tRequired step size is less than spacing between numbers."
                break
            elif not step_accepted:
                # Issue with step convergence
                status = -7
                message = "Error in step size calculation:\n\tError in step size acceptance."
                break

            # End of step loop. Update the _now variables
            t_old = t_now
            for i in range(y_size):
                y_old_ptr[i] = y_view[i]
                dy_old_ptr[i] = dy_ptr[i]

            # Store data
            if len_t >= current_size:
                # There is more data then we have room in our arrays.
                # Build new arrays with more space.
                # OPT: Note this is an expensive operation.
                num_concats += 1

                # Grow the array by 50% its current value
                current_size = <size_t> floor(<double>current_size * (1.5))

                time_domain_array_ptr = <double *> reallocate_mem(
                    time_domain_array_ptr,
                    current_size * sizeof(double),
                    'time_domain_array_ptr (growth stage)')

                y_results_array_ptr = <double_numeric *> reallocate_mem(
                    y_results_array_ptr,
                    y_size * current_size * sizeof(double_numeric),
                    'y_results_array_ptr (growth stage)')

                if capture_extra:
                    extra_array_ptr = <double_numeric *> reallocate_mem(
                        extra_array_ptr,
                        num_extra * current_size * sizeof(double_numeric),
                        'extra_array_ptr (growth stage)')

            # Add this step's results to our storage arrays.
            time_domain_array_ptr[len_t] = t_now
            for i in range(y_size):
                y_results_array_ptr[len_t * y_size + i] = y_view[i]

            if capture_extra:
                for i in range(num_extra):
                    extra_array_ptr[len_t * num_extra + i] = extra_output_ptr[i]

            # Increase number of independent variable points.
            len_t += 1

        # Integration has stopped. Check if it was successful.
        if status == 1:
            success = True
        else:
            success = False

        if success:
            # Solution was successful.

            # Load values into output arrays.
            # The arrays built during integration likely have a bunch of unused junk at the end due to overbuilding their size.
            # This process will remove that junk and leave only the valid data.
            # These arrays will always be the same length or less (self.len_t <= new_size) than the ones they are
            # built off of, so it is safe to use Realloc.
            solution_t_ptr = <double *> reallocate_mem(
                time_domain_array_ptr,
                len_t * sizeof(double),
                'solution_t_ptr (success stage)')
            time_domain_array_ptr = NULL

            solution_y_ptr = <double_numeric *> reallocate_mem(
                y_results_array_ptr,
                y_size * len_t * sizeof(double_numeric),
                'solution_y_ptr (success stage)')
            y_results_array_ptr = NULL

            if capture_extra:
                solution_extra_ptr = <double_numeric *> reallocate_mem(
                    extra_array_ptr,
                    num_extra * len_t * sizeof(double_numeric),
                    'solution_extra_ptr (success stage)')
                extra_array_ptr = NULL
        else:
            # Clear the storage arrays used during the step loop
            if not (time_domain_array_ptr is NULL):
                PyMem_Free(time_domain_array_ptr)
                time_domain_array_ptr = NULL
            if not (y_results_array_ptr is NULL):
                PyMem_Free(y_results_array_ptr)
                y_results_array_ptr = NULL
            if capture_extra:
                if not (extra_array_ptr is NULL):
                    PyMem_Free(extra_array_ptr)
                    extra_array_ptr = NULL

            # Integration was not successful. Leave the solution pointers as length 1 nan arrays.
            solution_t_ptr = <double *> allocate_mem(
                sizeof(double),
                'solution_t_ptr (fail stage)')
            solution_y_ptr = <double_numeric *> allocate_mem(
                y_size * sizeof(double_numeric),
                'solution_y_ptr (fail stage)')
            solution_extra_ptr = <double_numeric *> allocate_mem(
                num_extra * sizeof(double_numeric),
                'solution_extra_ptr (fail stage)')
            
            solution_t_ptr[0] = NAN
            for i in range(y_size):
                solution_y_ptr[i] = NAN
            for i in range(num_extra):
                solution_extra_ptr[i] = NAN

        # Integration is complete. Check if interpolation was requested.
        if success:
            if run_interpolation:
                # Use different len_t
                len_t_touse = len_t_eval
            else:
                len_t_touse = len_t
        else:
            # If integration was not successful use t_len = 1 to allow for nan arrays
            len_t_touse = 1

        if success and run_interpolation:
            # User only wants data at specific points.
            status = 2  # Interpolation is being performed.

            # TODO: The current version of cyrk_ode has not implemented sicpy's dense output. Instead we use an interpolation.
            # Build final interpolated time array
            interpolated_solution_t_ptr = <double *> allocate_mem(
                len_t_eval * sizeof(double),
                'interpolated_solution_t_ptr (interpolate stage)')

            # Build final interpolated solution arrays
            interpolated_solution_y_ptr = <double_numeric *> allocate_mem(
                y_size * len_t_eval * sizeof(double_numeric),
                'interpolated_solution_y_ptr (interpolate stage)')

            # Perform interpolation on y values
            interpolate(solution_t_ptr, t_eval_ptr, solution_y_ptr, interpolated_solution_y_ptr,
                        len_t, len_t_eval, y_size, y_is_complex)

            # Make a copy of t_eval (issues can arise if we store the t_eval pointer in solution array).
            for i in range(len_t_eval):
                interpolated_solution_t_ptr[i] = t_eval_ptr[i]

            if capture_extra:
                # Right now if there is any extra output then it is stored at each time step used in the RK loop.
                # We have to make a choice:
                #   - Do we interpolate the extra values that were stored?
                #   - Or do we use the interpolated t, y values to find new extra parameters at those specific points.
                # The latter method is more computationally expensive (recalls the diffeq for each y) but is more accurate.
                # This decision is set by the user with the `interpolate_extra` flag.

                # Build final interpolated solution array (Used if self.interpolate_extra is True or False)
                interpolated_solution_extra_ptr = <double_numeric *> allocate_mem(
                    num_extra * len_t_eval * sizeof(double_numeric),
                    'interpolated_solution_extra_ptr (interpolate)')

                if interpolate_extra:
                    # Perform interpolation on extra outputs
                    interpolate(solution_t_ptr, t_eval_ptr, solution_extra_ptr, interpolated_solution_extra_ptr,
                                len_t, len_t_eval, num_extra, y_is_complex)
                else:
                    # Use the new interpolated y and t values to recalculate the extra outputs with self.diffeq
                    for i in range(len_t_eval):
                        # Set state variables
                        t_now = t_eval_ptr[i]
                        for j in range(y_size):
                            y_view[j] = interpolated_solution_y_ptr[i * y_size + j]

                        # Call diffeq to recalculate extra outputs
                        if use_args:
                            diffeq(t_now, y_view, diffeq_out_view, *args)
                        else:
                            diffeq(t_now, y_view, diffeq_out_view)

                        # Capture extras
                        for j in range(num_extra):
                            interpolated_solution_extra_ptr[i * num_extra + j] = diffeq_out_view[extra_start + j]

                # Replace old pointers with new interpolated pointers and release the memory for the old stuff
                if not (solution_extra_ptr is NULL):
                    PyMem_Free(solution_extra_ptr)
                solution_extra_ptr = interpolated_solution_extra_ptr
                interpolated_solution_extra_ptr = NULL

            # Replace old pointers with new interpolated pointers and release the memory for the old stuff
            if not (solution_t_ptr is NULL):
                PyMem_Free(solution_t_ptr)
            solution_t_ptr = interpolated_solution_t_ptr
            interpolated_solution_t_ptr = NULL
            if not (solution_y_ptr is NULL):
                PyMem_Free(solution_y_ptr)
            solution_y_ptr = interpolated_solution_y_ptr
            interpolated_solution_y_ptr = NULL

            # Interpolation is done.
            status = 1

        # Convert solution pointers to a more user-friendly numpy ndarray
        solution_t = np.empty(len_t_touse, dtype=np.float64, order='C')
        solution_y = np.empty((total_size, len_t_touse), dtype=DTYPE, order='C')
        solution_t_view = solution_t
        solution_y_view = solution_y

        for i in range(len_t_touse):
            solution_t_view[i] = solution_t_ptr[i]
            for j in range(y_size):
                solution_y_view[j, i] = solution_y_ptr[i * y_size + j]
        if capture_extra:
            for i in range(len_t_touse):
                for j in range(num_extra):
                    solution_y_view[extra_start + j, i] = solution_extra_ptr[i * num_extra + j]
        # Free solution arrays
        if not (solution_t_ptr is NULL):
            PyMem_Free(solution_t_ptr)
            solution_t_ptr = NULL
        if not (solution_y_ptr is NULL):
            PyMem_Free(solution_y_ptr)
            solution_y_ptr = NULL
        if capture_extra:
            if not (solution_extra_ptr is NULL):
                PyMem_Free(solution_extra_ptr)
                solution_extra_ptr = NULL

        # Update integration message
        if status == 1:
            message = "Integration completed without issue."
        elif status == 0:
            message = "Integration is/was ongoing (perhaps it was interrupted?)."
        elif status == -1:
            message = "Error in step size calculation:\n\tRequired step size is less than spacing between numbers."
        elif status == -2:
            message = "Maximum number of steps (set by user) exceeded during integration."
        elif status == -3:
            message = "Maximum number of steps (set by system architecture) exceeded during integration."
        elif status == -6:
            message = "Integration never started: y-size is zero."
        elif status == -7:
            message = "Error in step size calculation:\n\tError in step size acceptance."

    finally:
        # Free pointers made from user inputs
        if not (tol_ptrs is NULL):
            PyMem_Free(tol_ptrs)
            tol_ptrs = NULL
        if not (t_eval_ptr is NULL):
            PyMem_Free(t_eval_ptr)
            t_eval_ptr = NULL

        # Free pointers used to track y, dydt, and any extra outputs
        if not (y_storage_ptrs is NULL):
            PyMem_Free(y_storage_ptrs)
            y_storage_ptrs = NULL
        if not (extra_output_init_ptr is NULL):
            PyMem_Free(extra_output_init_ptr)
            extra_output_init_ptr = NULL
        if not (extra_output_ptr is NULL):
            PyMem_Free(extra_output_ptr)
            extra_output_ptr = NULL

        # Free RK Temp Storage Array
        if not (K_ptr is NULL):
            PyMem_Free(K_ptr)
            K_ptr = NULL

        # Free other pointers that should have been freed in main loop, but in case of an exception they were missed.
        if not (solution_t_ptr is NULL):
            PyMem_Free(solution_t_ptr)
            solution_t_ptr = NULL
        if not (solution_y_ptr is NULL):
            PyMem_Free(solution_y_ptr)
            solution_y_ptr = NULL
        if not (solution_extra_ptr is NULL):
            PyMem_Free(solution_extra_ptr)
            solution_extra_ptr = NULL
        if not (interpolated_solution_t_ptr is NULL):
            PyMem_Free(interpolated_solution_t_ptr)
            interpolated_solution_t_ptr= NULL
        if not (interpolated_solution_y_ptr is NULL):
            PyMem_Free(interpolated_solution_y_ptr)
            interpolated_solution_y_ptr = NULL
        if not (interpolated_solution_extra_ptr is NULL):
            PyMem_Free(interpolated_solution_extra_ptr)
            interpolated_solution_extra_ptr = NULL
        if not (time_domain_array_ptr is NULL):
            PyMem_Free(time_domain_array_ptr)
            time_domain_array_ptr = NULL
        if not (y_results_array_ptr is NULL):
            PyMem_Free(y_results_array_ptr)
            y_results_array_ptr = NULL
        if not (extra_array_ptr is NULL):
            PyMem_Free(extra_array_ptr)
            extra_array_ptr = NULL

    return solution_t, solution_y, success, message
