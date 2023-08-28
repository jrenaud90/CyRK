# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import cython
import sys
import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool as bool_cpp_t
from libc.math cimport sqrt, fabs, nextafter, fmax, fmin

from CyRK.array.interp cimport interp_array, interp_complex_array
from CyRK.rk.rk cimport (
    RK23_C, RK23_B, RK23_E, RK23_A, RK23_order, RK23_error_order, RK23_n_stages, RK23_LEN_C, RK23_LEN_B, RK23_LEN_E,
    RK23_LEN_E3, RK23_LEN_E5, RK23_LEN_A0, RK23_LEN_A1,
    RK45_C, RK45_B, RK45_E, RK45_A, RK45_order, RK45_error_order, RK45_n_stages, RK45_LEN_C, RK45_LEN_B, RK45_LEN_E,
    RK45_LEN_E3, RK45_LEN_E5, RK45_LEN_A0, RK45_LEN_A1,
    DOP_C_REDUCED, DOP_B, DOP_E3, DOP_E5, DOP_A_REDUCED, DOP_order, DOP_error_order, DOP_n_stages,
    DOP_n_stages_extended, DOP_LEN_C, DOP_LEN_B, DOP_LEN_E, DOP_LEN_E3, DOP_LEN_E5, DOP_LEN_A0, DOP_LEN_A1)

# # Integration Constants
# Multiply steps computed from asymptotic behaviour of errors by this.
cdef double SAFETY = 0.9
cdef double MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
cdef double MAX_FACTOR = 10.  # Maximum allowed increase in a step size.
cdef double MAX_STEP = np.inf
cdef double INF = np.inf
cdef double EPS = np.finfo(np.float64).eps
cdef double EPS_10 = EPS * 10.
cdef double EPS_100 = EPS * 100.
cdef Py_ssize_t MAX_INT_SIZE = int(0.95 * sys.maxsize)


cdef double cabs(double complex value) nogil:  # REVERT noexcept nogil:
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

# Define fused type to handle both float and complex-valued versions of y and dydt.
ctypedef fused double_numeric:
    double
    double complex


cdef double dabs(double_numeric value) nogil:  # REVERT noexcept nogil:
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
    bool_cpp_t capture_extra = False,
    Py_ssize_t num_extra = 0,
    bool_cpp_t interpolate_extra = False,
    Py_ssize_t expected_size = 0,
    Py_ssize_t max_num_steps = 0
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
    max_num_steps : Py_ssize_t, default=0
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
    interpolate_extra : bool_cpp_t, default=False
        Flag if interpolation should be run on extra parameters.
        If set to False when `run_interpolation=True`, then interpolation will be run on solution's y, t. These will
        then be used to recalculate extra parameters rather than an interpolation on the extra parameters captured
        during integration.
    expected_size : Py_ssize_t, default=0
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
    cdef Py_ssize_t s, i, j

    # Setup integration variables
    cdef char status, old_status
    cdef str message

    # Determine information about the differential equation based on its initial conditions
    cdef Py_ssize_t y_size
    cdef double y_size_dbl, y_size_sqrt
    cdef bool_cpp_t y_is_complex
    y_size = y0.size
    y_is_complex = False
    y_size_dbl = <double>y_size
    y_size_sqrt = sqrt(y_size_dbl)

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
    cdef double t_start, t_end, t_delta, t_delta_check, t_delta_abs, direction_inf, t_old, t_new, time_
    cdef bool_cpp_t direction_flag
    t_start = t_span[0]
    t_end   = t_span[1]
    t_delta = t_end - t_start
    t_delta_abs = fabs(t_delta)
    t_delta_check = t_delta_abs
    if t_delta >= 0.:
        # Integration is moving forward in time.
        direction_flag = True
        direction_inf = INF
    else:
        # Integration is moving backwards in time.
        direction_flag = False
        direction_inf = -INF

    # Pull out information on t-eval
    cdef Py_ssize_t len_teval
    if t_eval is None:
        len_teval = 0
    else:
        len_teval = t_eval.size

    # Pull out information on args
    cdef bool_cpp_t use_args
    if args is None:
        use_args = False
    else:
        use_args = True

    # Set integration flags
    cdef bool_cpp_t success, step_accepted, step_rejected, step_error, run_interpolation, \
        store_extras_during_integration
    success           = False
    step_accepted     = False
    step_rejected     = False
    step_error        = False
    run_interpolation = False
    store_extras_during_integration = capture_extra
    if len_teval > 0:
        run_interpolation = True
    if run_interpolation and not interpolate_extra:
        # If y is eventually interpolated but the extra outputs are not being interpolated, then there is
        #  no point in storing the values during the integration. Turn off this functionality to save
        #  on computation.
        store_extras_during_integration = False

    # # Determine integration tolerances
    use_arg_arrays = False
    use_atol_array = False
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] rtol_array, atol_array
    rtol_array = np.empty(y_size, dtype=np.float64, order='C')
    atol_array = np.empty(y_size, dtype=np.float64, order='C')
    cdef double[::1] rtols_view, atols_view
    rtols_view = rtol_array
    atols_view = atol_array

    if rtols is not None:
        # Using arrayed rtol
        if len(rtols) != y_size:
            raise AttributeError('rtols must be the same size as y0.')
        for i in range(y_size):
            rtol = rtols[i]
            if rtol < EPS_100:
                rtol = EPS_100
            rtols_view[i] = rtol
    else:
        # Using constant rtol
        # Check tolerances
        if rtol < EPS_100:
            rtol = EPS_100
        for i in range(y_size):
            rtols_view[i] = rtol

    if atols is not None:
        # Using arrayed atol
        if len(atols) != y_size:
            raise AttributeError('atols must be the same size as y0.')
        for i in range(y_size):
            atols_view[i] = atols[i]
    else:
        for i in range(y_size):
            atols_view[i] = atol

    # Determine maximum number of steps
    if max_num_steps == 0:
        max_num_steps = MAX_INT_SIZE
    elif max_num_steps < 0:
        raise AttributeError('Negative number of max steps provided.')
    else:
        max_num_steps = min(max_num_steps, MAX_INT_SIZE)

    # Expected size of output arrays.
    cdef double temp_expected_size
    cdef Py_ssize_t expected_size_to_use, num_concats
    if expected_size == 0:
        # CySolver will attempt to guess the best size for the output arrays.
        temp_expected_size = 100. * t_delta_abs * fmax(1., (1.e-6 / rtol))
        temp_expected_size = fmax(temp_expected_size, 100.)
        temp_expected_size = fmin(temp_expected_size, 10_000_000.)
        expected_size_to_use = <Py_ssize_t>temp_expected_size
    else:
        expected_size_to_use = <Py_ssize_t>expected_size
    # This variable tracks how many times the storage arrays have been appended.
    # It starts at 1 since there is at least one storage array present.
    num_concats = 1

    # Initialize arrays that are based on y's size and type.
    y_new    = np.empty(y_size, dtype=DTYPE, order='C')
    y_old    = np.empty(y_size, dtype=DTYPE, order='C')
    dydt_new = np.empty(y_size, dtype=DTYPE, order='C')
    dydt_old = np.empty(y_size, dtype=DTYPE, order='C')

    # Setup memory views for these arrays
    cdef double_numeric[:] y_new_view, y_old_view, dydt_new_view, dydt_old_view
    y_new_view    = y_new
    y_old_view    = y_old
    dydt_new_view = dydt_new
    dydt_old_view = dydt_old

    # Store y0 into the y arrays
    cdef double_numeric y_value
    for i in range(y_size):
        y_value = y0[i]
        y_new_view[i] = y_value
        y_old_view[i] = y_value

    # If extra output is true then the output of the diffeq will be larger than the size of y0.
    # Determine that extra size by calling the diffeq and checking its size.
    cdef Py_ssize_t extra_start, total_size, store_loop_size
    extra_start = y_size
    total_size  = y_size + num_extra
    # Create arrays based on this total size
    diffeq_out    = np.empty(total_size, dtype=DTYPE, order='C')
    y0_plus_extra = np.empty(total_size, dtype=DTYPE, order='C')
    extra_result  = np.empty(num_extra, dtype=DTYPE, order='C')

    # Setup memory views
    cdef double_numeric[:] diffeq_out_view, y0_plus_extra_view, extra_result_view
    diffeq_out_view    = diffeq_out
    y0_plus_extra_view = y0_plus_extra
    extra_result_view  = extra_result

    # Capture the extra output for the initial condition.
    if capture_extra:
        if use_args:
            diffeq(t_start, y_new, diffeq_out, *args)
        else:
            diffeq(t_start, y_new, diffeq_out)

        # Extract the extra output from the function output.
        for i in range(total_size):
            if i < extra_start:
                # Pull from y0
                y0_plus_extra_view[i] = y0[i]
            else:
                # Pull from extra output
                y0_plus_extra_view[i] = diffeq_out_view[i]
        if store_extras_during_integration:
            store_loop_size = total_size
        else:
            store_loop_size = y_size
    else:
        # No extra output
        store_loop_size = y_size

    y0_to_store = np.empty(store_loop_size, dtype=DTYPE, order='C')
    cdef double_numeric[:] y0_to_store_view
    y0_to_store_view = y0_to_store
    for i in range(store_loop_size):
        if store_extras_during_integration:
            y0_to_store_view[i] = y0_plus_extra_view[i]
        else:
            y0_to_store_view[i] = y0[i]

    # # Determine RK scheme
    cdef Py_ssize_t rk_order, error_order
    cdef Py_ssize_t rk_n_stages, rk_n_stages_plus1, rk_n_stages_extended
    cdef Py_ssize_t len_C, len_B, len_E, len_E3, len_E5, len_A0, len_A1
    cdef double error_pow, error_expo, error_norm5, error_norm3, error_norm, error_norm_abs, error_norm3_abs, error_norm5_abs, error_denom

    if rk_method == 0:
        # RK23 Method
        rk_order    = RK23_order
        error_order = RK23_error_order
        rk_n_stages = RK23_n_stages
        len_C       = RK23_LEN_C
        len_B       = RK23_LEN_B
        len_E       = RK23_LEN_E
        len_E3      = RK23_LEN_E3
        len_E5      = RK23_LEN_E5
        len_A0      = RK23_LEN_A0
        len_A1      = RK23_LEN_A1
    elif rk_method == 1:
        # RK45 Method
        rk_order    = RK45_order
        error_order = RK45_error_order
        rk_n_stages = RK45_n_stages
        len_C       = RK45_LEN_C
        len_B       = RK45_LEN_B
        len_E       = RK45_LEN_E
        len_E3      = RK45_LEN_E3
        len_E5      = RK45_LEN_E5
        len_A0      = RK45_LEN_A0
        len_A1      = RK45_LEN_A1
    elif rk_method == 2:
        # DOP853 Method
        rk_order    = DOP_order
        error_order = DOP_error_order
        rk_n_stages = DOP_n_stages
        len_C       = DOP_LEN_C
        len_B       = DOP_LEN_B
        len_E       = DOP_LEN_E
        len_E3      = DOP_LEN_E3
        len_E5      = DOP_LEN_E5
        len_A0      = DOP_LEN_A0
        len_A1      = DOP_LEN_A1

        rk_n_stages_extended = DOP_n_stages_extended
    else:
        status = -8
        message = "Attribute error."
        raise AttributeError(
            'Unexpected rk_method provided. Currently supported versions are:\n'
            '\t0 = RK23\n'
            '\t1 = RK34\n'
            '\t2 = DOP853')

    rk_n_stages_plus1 = rk_n_stages + 1
    error_expo = 1. / (<double>error_order + 1.)

    # Build RK Arrays. Note that all are 1D except for A and K.
    A      = np.empty((len_A0, len_A1), dtype=DTYPE, order='C')
    B      = np.empty(len_B, dtype=DTYPE, order='C')
    C      = np.empty(len_C, dtype=np.float64, order='C')  # C is always float no matter what y0 is.
    E      = np.empty(len_E, dtype=DTYPE, order='C')
    E3     = np.empty(len_E3, dtype=DTYPE, order='C')
    E5     = np.empty(len_E5, dtype=DTYPE, order='C')
    K      = np.zeros((rk_n_stages_plus1, y_size), dtype=DTYPE, order='C')  # It is important K be initialized with 0s

    # Setup memory views.
    cdef double_numeric[:] B_view, E_view, E3_view, E5_view
    cdef double_numeric[:, :] A_view, K_view
    cdef double_numeric A_at_sj, B_at_j, error_dot_1, error_dot_2
    cdef double[:] C_view
    A_view      = A
    B_view      = B
    C_view      = C
    E_view      = E
    E3_view     = E3
    E5_view     = E5
    K_view      = K

    # Populate values based on externally defined constants.
    if rk_method == 0:
        # RK23 Method
        for i in range(len_A0):
            for j in range(len_A1):
                A_view[i, j] = RK23_A[i][j]
        for i in range(len_B):
            B_view[i] = RK23_B[i]
        for i in range(len_C):
            C_view[i] = RK23_C[i]
        for i in range(len_E):
            E_view[i] = RK23_E[i]
            # Dummy Variables, set equal to E
            E3_view[i] = RK23_E[i]
            E5_view[i] = RK23_E[i]
    elif rk_method == 1:
        # RK45 Method
        for i in range(len_A0):
            for j in range(len_A1):
                A_view[i, j] = RK45_A[i][j]
        for i in range(len_B):
            B_view[i] = RK45_B[i]
        for i in range(len_C):
            C_view[i] = RK45_C[i]
        for i in range(len_E):
            E_view[i] = RK45_E[i]
            # Dummy Variables, set equal to E
            E3_view[i] = RK45_E[i]
            E5_view[i] = RK45_E[i]
    else:
        # DOP853 Method
        for i in range(len_A0):
            for j in range(len_A1):
                A_view[i, j] = DOP_A_REDUCED[i][j]
        for i in range(len_B):
            B_view[i] = DOP_B[i]
        for i in range(len_C):
            C_view[i] = DOP_C_REDUCED[i]
        for i in range(len_E):
            E3_view[i] = DOP_E3[i]
            E5_view[i] = DOP_E5[i]
            E_view[i] = DOP_E5[i]
            # Dummy Variables, set equal to E3
            E_view[i] = DOP_E3[i]

    # Initialize variables for start of integration
    if not capture_extra:
        # If `capture_extra` is True then this step was already performed.
        if use_args:
            diffeq(t_start, y_new, diffeq_out, *args)
        else:
            diffeq(t_start, y_new, diffeq_out)

    t_old = t_start
    t_new = t_start
    # Initialize dydt arrays.
    for i in range(y_size):
        dydt_new_view[i] = diffeq_out_view[i]
        dydt_old_view[i] = dydt_new_view[i]
    
    # Setup storage arrays
    # These arrays are built to fit a number of points equal to `expected_size_to_use`
    # If the integration needs more than that then a new array will be concatenated (with performance costs) to these.
    cdef double_numeric[:, :] y_results_array_view, y_results_array_new_view, solution_y_view
    cdef double[:] time_domain_array_view, time_domain_array_new_view, solution_t_view
    y_results_array        = np.empty((store_loop_size, expected_size_to_use), dtype=DTYPE, order='C')
    time_domain_array      = np.empty(expected_size_to_use, dtype=np.float64, order='C')
    y_results_array_view   = y_results_array
    time_domain_array_view = time_domain_array

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] scale_arr
    cdef double[:] scale_view
    cdef double scale
    scale_arr = np.empty(y_size, dtype=np.float64, order='C')
    scale_view = scale_arr

    # Load initial conditions into output arrays
    time_domain_array_view[0] = t_start
    for i in range(store_loop_size):
        if store_extras_during_integration:
            y_results_array_view[i] = y0_plus_extra_view[i]
        else:
            y_results_array_view[i] = y0[i]

    # # Determine size of first step.
    cdef double step_size, d0, d1, d2, d0_abs, d1_abs, d2_abs, h0, h1
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

                scale = atols_view[i] + dabs(y_old_view[i]) * rtols_view[i]
                d0_abs = dabs(y_old_view[i] / scale)
                d1_abs = dabs(dydt_old_view[i] / scale)
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
            t_new = t_old + h0_direction
            for i in range(y_size):
                y_new_view[i] = y_old_view[i] + h0_direction * dydt_old_view[i]

            if use_args:
                diffeq(t_new, y_new, diffeq_out, *args)
            else:
                diffeq(t_new, y_new, diffeq_out)

            # Find the norm for d2
            d2 = 0.
            for i in range(y_size):
                dydt_new_view[i] = diffeq_out_view[i]
                scale = atols_view[i] + dabs(y_old_view[i]) * rtols_view[i]
                d2_abs = dabs( (dydt_new_view[i] - dydt_old_view[i]) / scale)
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
    cdef double min_step, step_factor, step
    cdef double c
    cdef double_numeric K_scale
    cdef Py_ssize_t len_t
    status = 0
    message = "Integration is/was ongoing (perhaps it was interrupted?)."
    len_t  = 1  # There is an initial condition provided so the time length is already 1

    if y_size == 0:
        status = -6
        message = "Integration never started: y-size is zero."

    while status == 0:
        if t_new == t_end:
            t_old = t_end
            status = 1
            break

        if len_t > max_num_steps:
            if max_num_steps == MAX_INT_SIZE:
                status = -3
                message = "Maximum number of steps (set by system architecture) exceeded during integration."
            else:
                status = -2
                message = "Maximum number of steps (set by user) exceeded during integration."
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
                t_delta_check = t_new - t_end
            else:
                step = -step_size
                t_delta_check = t_end - t_new
            t_new = t_old + step

            # Check that we are not at the end of integration with that move
            if t_delta_check > 0.:
                t_new = t_end

                # Correct the step if we were at the end of integration
                step = t_new - t_old
                if direction_flag:
                    step_size = step
                else:
                    step_size = -step

            # Calculate derivative using RK method
            for i in range(y_size):
                K_view[0, i] = dydt_old_view[i]

            for s in range(1, len_C):
                c = C_view[s]
                time_ = t_old + c * step

                # Dot Product (K, a) * step
                for j in range(s):
                    A_at_sj = A_view[s, j]
                    for i in range(y_size):
                        if j == 0:
                            # Initialize
                            y_new_view[i] = y_old_view[i]

                        y_new_view[i] = y_new_view[i] + (K_view[j, i] * A_at_sj * step)

                if use_args:
                    diffeq(time_, y_new, diffeq_out, *args)
                else:
                    diffeq(time_, y_new, diffeq_out)

                for i in range(y_size):
                    K_view[s, i] = diffeq_out_view[i]

            # Dot Product (K, B) * step
            for j in range(rk_n_stages):
                B_at_j = B_view[j]
                # We do not use rk_n_stages_plus1 here because we are chopping off the last row of K to match
                #  the shape of B.
                for i in range(y_size):
                    if j == 0:
                        # Initialize
                        y_new_view[i] = y_old_view[i]
                    y_new_view[i] = y_new_view[i] + (K_view[j, i] * B_at_j * step)

            if use_args:
                diffeq(t_new, y_new, diffeq_out, *args)
            else:
                diffeq(t_new, y_new, diffeq_out)


            for i in range(store_loop_size):
                if i < extra_start:
                    # Set diffeq results
                    dydt_new_view[i] = diffeq_out_view[i]
                    scale_view[i] = atols_view[i] + max(dabs(y_old_view[i]), dabs(y_new_view[i])) * rtols_view[i]

                    # Set last array of K equal to dydt
                    K_view[rk_n_stages, i] = dydt_new_view[i]

                else:
                    # Set extra results
                    extra_result_view[i - extra_start] = diffeq_out_view[i]

            if rk_method == 2:
                # Calculate Error for DOP853
                # Find norms for each error
                error_norm5 = 0.
                error_norm3 = 0.
                # Dot Product (K, E5) / scale and Dot Product (K, E3) * step / scale
                for i in range(y_size):
                    for j in range(rk_n_stages_plus1):
                        if j == 0:
                            # Initialize
                            error_dot_1 = 0.
                            error_dot_2 = 0.

                        scale = scale_view[i]
                        K_scale = K_view[j, i] / <double_numeric>scale
                        error_dot_1 += K_scale * E3_view[j]
                        error_dot_2 += K_scale * E5_view[j]

                    error_norm3_abs = dabs(error_dot_1)
                    error_norm5_abs = dabs(error_dot_2)

                    error_norm3 += (error_norm3_abs * error_norm3_abs)
                    error_norm5 += (error_norm5_abs * error_norm5_abs)

                # Check if errors are zero
                if (error_norm5 == 0.) and (error_norm3 == 0.):
                    error_norm = 0.
                else:
                    error_denom = error_norm5 + 0.01 * error_norm3
                    error_norm = step_size * error_norm5 / sqrt(error_denom * y_size_dbl)

            else:
                # Calculate Error for RK23 and RK45
                # Dot Product (K, E) * step / scale
                error_norm = 0.
                for i in range(y_size):
                    for j in range(rk_n_stages_plus1):
                        if j == 0:
                            # Initialize
                            error_dot_1 = 0.

                        scale = scale_view[i]
                        K_scale = K_view[j, i] / <double_numeric> scale
                        error_dot_1 += K_scale * E_view[j] * step

                    error_norm_abs = dabs(error_dot_1)
                    error_norm += (error_norm_abs * error_norm_abs)
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
        t_old = t_new
        for i in range(y_size):
            y_old_view[i] = y_new_view[i]
            dydt_old_view[i] = dydt_new_view[i]

        # Save data
        if len_t >= (num_concats * expected_size_to_use):                
            # There is more data than we have room in our arrays. 
            # Build new arrays with more space.
            # OPT: Note this is an expensive operation. 
            num_concats += 1
            new_size = num_concats * expected_size_to_use
            time_domain_array_new      = np.empty(new_size, dtype=np.float64, order='C')
            y_results_array_new        = np.empty((store_loop_size, new_size), dtype=DTYPE, order='C')
            time_domain_array_new_view = time_domain_array_new
            y_results_array_new_view   = y_results_array_new
            
            # Loop through time to fill in these new arrays with the old values
            for i in range(len_t):
                time_domain_array_new_view[i] = time_domain_array_view[i]
                
                for j in range(store_loop_size):
                    y_results_array_new_view[j, i] = y_results_array_view[j, i]
            
            # No longer need the old arrays. Change where the view is pointing and delete them.
            y_results_array_view = y_results_array_new
            time_domain_array_view = time_domain_array_new
            # TODO: Delete the old arrays?
        
        # There should be room in the arrays to add new data.
        time_domain_array_view[len_t] = t_new
        # To match the format that scipy follows, we will take the transpose of y.
        for i in range(store_loop_size):
            if i < extra_start:
                # Pull from y result
                y_results_array_view[i, len_t] = y_new_view[i]
            else:
                # Pull from extra
                y_results_array_view[i, len_t] = extra_result_view[i - extra_start]

        # Increase number of time points.
        len_t += 1

    # # Clean up output.
    if status == 1:
        success = True
        message = "Integration completed without issue."

    # Create output arrays. To match the format that scipy follows, we will take the transpose of y.
    if success:
        # Build final output arrays.
        # The arrays built during integration likely have a bunch of unused junk at the end due to overbuilding their size.
        # This process will remove that junk and leave only the wanted data.
        solution_y = np.empty((store_loop_size, len_t), dtype=DTYPE, order='C')
        solution_t = np.empty(len_t, dtype=np.float64, order='C')

        # Link memory views
        solution_y_view = solution_y
        solution_t_view = solution_t

        # Populate values
        for i in range(len_t):
            solution_t_view[i] = time_domain_array_view[i]
            for j in range(store_loop_size):
                solution_y_view[j, i] = y_results_array_view[j, i]
    else:
        # Build nan arrays
        solution_y = np.nan * np.ones((store_loop_size, 1), dtype=DTYPE, order='C')
        solution_t = np.nan * np.ones(1, dtype=np.float64, order='C')

        # Link memory views
        solution_y_view = solution_y
        solution_t_view = solution_t

    cdef double_numeric[:, :] y_results_reduced_view
    cdef double_numeric[:] y_result_timeslice_view, y_result_temp_view, y_interp_view

    if run_interpolation and success:
        old_status = status
        status = 2
        # User only wants data at specific points.

        # The current version of this function has not implemented sicpy's dense output.
        #   Instead we use an interpolation.
        # OPT: this could be done inside the integration loop for performance gains.
        y_results_reduced       = np.empty((total_size, len_teval), dtype=DTYPE, order='C')
        y_result_timeslice      = np.empty(len_t, dtype=DTYPE, order='C')
        y_result_temp           = np.empty(len_teval, dtype=DTYPE, order='C')
        y_results_reduced_view  = y_results_reduced
        y_result_timeslice_view = y_result_timeslice
        y_result_temp_view      = y_result_temp

        for j in range(y_size):
            # np.interp only works on 1D arrays so we must loop through each of the variables:
            # # Set timeslice equal to the time values at this y_j
            for i in range(len_t):
                y_result_timeslice_view[i] = solution_y_view[j, i]

            # Perform numerical interpolation
            if double_numeric is cython.doublecomplex:
                interp_complex_array(
                    t_eval,
                    solution_t_view,
                    y_result_timeslice_view,
                    y_result_temp_view
                    )
            else:
                interp_array(
                    t_eval,
                    solution_t_view,
                    y_result_timeslice_view,
                    y_result_temp_view
                    )

            # Store result.
            for i in range(len_teval):
                y_results_reduced_view[j, i] = y_result_temp_view[i]

        if capture_extra:
            # Right now if there is any extra output then it is stored at each time step used in the RK loop.
            # We have to make a choice on what to output do we, like we do with y, interpolate all of those extras?
            #  or do we use the interpolation on y to find new values.
            # The latter method is more computationally expensive (recalls the diffeq for each y) but is more accurate.
            if interpolate_extra:
                # Continue the interpolation for the extra values.
                for j in range(num_extra):
                    # np.interp only works on 1D arrays so we must loop through each of the variables:
                    # # Set timeslice equal to the time values at this y_j
                    for i in range(len_t):
                        y_result_timeslice_view[i] = solution_y_view[extra_start + j, i]

                    # Perform numerical interpolation
                    if double_numeric is cython.doublecomplex:
                        interp_complex_array(
                                t_eval,
                                solution_t_view,
                                y_result_timeslice_view,
                                y_result_temp_view
                                )
                    else:
                        interp_array(
                                t_eval,
                                solution_t_view,
                                y_result_timeslice_view,
                                y_result_temp_view
                                )

                    # Store result.
                    for i in range(len_teval):
                        y_results_reduced_view[extra_start + j, i] = y_result_temp_view[i]
            else:
                # Use y and t to recalculate the extra outputs
                y_interp = np.empty(y_size, dtype=DTYPE)
                y_interp_view = y_interp
                for i in range(len_teval):
                    time_ = t_eval[i]
                    for j in range(y_size):
                        y_interp_view[j] = y_results_reduced_view[j, i]

                    if use_args:
                        diffeq(time_, y_interp, diffeq_out, *args)
                    else:
                        diffeq(time_, y_interp, diffeq_out)

                    for j in range(num_extra):
                        y_results_reduced_view[extra_start + j, i] = diffeq_out_view[extra_start + j]

        # Replace the output y results and time domain with the new reduced one
        solution_y = np.empty((total_size, len_teval), dtype=DTYPE, order='C')
        solution_t = np.empty(len_teval, dtype=np.float64, order='C')
        solution_y_view = solution_y
        solution_t_view = solution_t

        # Update output arrays
        for i in range(len_teval):
            solution_t_view[i] = t_eval[i]
            for j in range(total_size):
                # To match the format that scipy follows, we will take the transpose of y.
                solution_y_view[j, i] = y_results_reduced_view[j, i]
        status = old_status

    return solution_t, solution_y, success, message
