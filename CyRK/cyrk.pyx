import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cpython cimport array
np.import_array()

# Runge-Kutta Parameters
# Multiply steps computed from asymptotic behaviour of errors by this.
cdef double SAFETY = 0.9

cdef double MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
cdef double MAX_FACTOR = 10.  # Maximum allowed increase in a step size.

# System / Computation Parameters
cdef double EPS = np.finfo(np.float64).eps
cdef double EPS_10 = 10. * EPS
cdef double EPS_100 = 100. * EPS
cdef double INF = np.inf
cdef double MAX_STEP = np.inf

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double _norm(double complex[:] x):
    cdef double size_sqrt, value_abs
    cdef unsigned int x_isize = x.shape[0]
    cdef int i
    cdef double x_size, frobenius_sum
    
    # Initialize summation
    frobenius_sum = 0.
    
    # Find sqrt(size)
    x_size = <double>x_isize
    size_sqrt = sqrt(x_size)
    
    # Perform summation
    for i in range(x_isize):
        value_abs = abs(x[i])
        frobenius_sum += value_abs * value_abs
    
    # Find sqrt-norm
    result = sqrt(frobenius_sum) / size_sqrt
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
def cyrk_ode(
    diffeq,
    (double, double) t_span, 
    np.ndarray[np.complex128_t, ndim=1] y0, 
    tuple args = tuple(),
    double rtol = 1.e-6, 
    double atol = 1.e-8,
    double max_step = MAX_STEP, 
    double first_step = 0.,
    unsigned int rk_method = 1
    ):
    """ A Numba-safe Rugge-Kutta Integrator based on Scipy's solve_ivp RK integrator.

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
            # TODO NotImplemented 2 = DOP
    t_eval : np.ndarray = None
        If provided, then the function will interpolate the integration results to provide them at the
            requested t-steps.

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
    cdef bint success = False
    cdef bint step_accepted = False
    cdef bint step_rejected = False
    cdef bint step_error = False
    cdef unsigned int s, rk_order, error_order, rk_n_stages, y_size, rk_n_stages_plus1, len_c
    cdef int status, j, i
    cdef double t_start, t_end, t_delta, t_delta_abs, t_init_step
    cdef double direction, error_expo, error_norm, h0_direction
    cdef double d0, d1, h0, h1, step_size, time_, min_step, step_factor, c
    cdef np.ndarray[np.float64_t, ndim=1] C, scale
    cdef np.ndarray[np.complex128_t, ndim=1] y_new, y_old, y_tmp, dydt_new, dydt_old, dydt_init_step, y_init_step
    cdef np.ndarray[np.complex128_t, ndim=1] y_now_ovr_scale, dydt_now_ovr_scale
    cdef np.ndarray[np.complex128_t, ndim=2] y_results
    cdef np.ndarray[np.complex128_t, ndim=1] B, E, K_tmp
    cdef np.ndarray[np.complex128_t, ndim=2] A, K
    cdef list time_domain_list, y_results_list
    cdef str message, rejected_message
    cdef double[:] scale_view, time_domain_view
    cdef double complex[:] y_old_view, dydt_old_view
    cdef double complex[:] y_now_ovr_scale_view, dydt_now_ovr_scale_view, dydt_init_step_view
    cdef double complex[:, :] K_view, A_view
    cdef double complex[:] E_view, B_view, K_tmp_view, A_tmp_view
    cdef double complex[:, :] y_results_T_view, y_results_view

    # Clean up and interpret inputs
    # Time Domain
    t_start = t_span[0]
    t_end = t_span[1]
    t_delta = t_end - t_start
    t_delta_abs = abs(t_delta)
    if t_delta >= 0.:
        direction = 1.
    else:
        direction = -1.
    
    # y-array
    y0 = np.asarray(y0)
    y_size = y0.size
    dtype = y0.dtype
    
    # Start storing results with the initial conditions
    time_domain_list = [t_start]
    y_results_list = [y0]
    
    # Initialize arrays
    scale              = np.empty(y_size, dtype=np.float64)
    y_now_ovr_scale    = np.empty(y_size, dtype=dtype)
    dydt_now_ovr_scale = np.empty(y_size, dtype=dtype)
    y_init_step        = np.empty(y_size, dtype=dtype)
    y_new              = np.empty(y_size, dtype=dtype)
    y_old              = np.empty(y_size, dtype=dtype)
    dydt_new           = np.empty(y_size, dtype=dtype)
    dydt_old           = np.empty(y_size, dtype=dtype)
    dydt_init_step     = np.empty(y_size, dtype=dtype)
    y_tmp              = np.empty(y_size, dtype=dtype)

    # Integrator Status Codes
    #   0  = Running
    #   -1 = Failed
    #   1  = Finished with no obvious issues
    status = 0

    # Determine RK constants
    if rk_method == 0:
        # RK23 Method
        rk_order = 3
        error_order = 2
        rk_n_stages = 3
        C = np.array([0., 1. / 2., 3. / 4.], dtype=np.float64)
        A = np.array(
            [
                [0., 0., 0.],
                [1. / 2., 0., 0.],
                [0., 3. / 4., 0.]
                ],
            order='C', dtype=np.complex128
            )
        B = np.array([2. / 9., 1. / 3., 4. / 9.],
                     order='C', dtype=np.complex128)
        E = np.array([5. / 72., -1. / 12., -1. / 9., 1. / 8.],
                     order='C', dtype=np.complex128)

        len_c = 3
    else:
        # RK45 Method
        rk_order = 5
        error_order = 4
        rk_n_stages = 6        
        C = np.array([0., 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1.], dtype=np.float64)
        A = np.array(
            [
                [0., 0., 0., 0., 0.],
                [1. / 5., 0., 0., 0., 0.],
                [3. / 40., 9. / 40., 0., 0., 0.],
                [44. / 45., -56. / 15., 32. / 9., 0., 0.],
                [19372. / 6561., -25360. / 2187., 64448. / 6561., -212. / 729., 0.],
                [9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176., -5103. / 18656.]
                ], order='C', dtype=np.complex128
            )
        B = np.array([35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84.],
                     order='C', dtype=np.complex128)
        E = np.array(
            [-71. / 57600., 0., 71. / 16695., -71. / 1920., 17253. / 339200., -22. / 525., 1. / 40.],
            order='C', dtype=np.complex128
            )
        
        len_c = 6
    
    # Set memoryviews
    A_view                  = A
    E_view                  = E
    B_view                  = B
    scale_view              = scale
    y_old_view              = y_old
    dydt_old_view           = dydt_old
    y_now_ovr_scale_view    = y_now_ovr_scale
    dydt_now_ovr_scale_view = dydt_now_ovr_scale
    dydt_init_step_view     = dydt_init_step

    error_expo = 1. / (<double>error_order + 1.)

    # Check tolerances
    if rtol < EPS_100:
        rtol = EPS_100

#     atol_arr = np.asarray(atol, dtype=np.complex128)
#     if atol_arr.ndim > 0 and atol_arr.shape[0] != y_size:
#         # atol must be either the same for all y or must be provided as an array, one for each y.
#         raise Exception

    # Initialize variables for start of integration
    diffeq(t_start, y0, dydt_new, *args)
    t_old = t_start
    t_new = t_start
    for i in range(y_size):
        dydt_old[i] = dydt_new[i]
        y_old[i] = y0[i]
        y_new[i] = y0[i]

    if first_step == 0.:
        # Select an initial step size based on the differential equation.
        # .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
        #        Equations I: Nonstiff Problems", Sec. II.4.
        if y_size == 0:
            step_size = INF
        else:
            for i in range(y_size):
                scale[i] = atol + abs(y_old_view[i]) * rtol
                y_now_ovr_scale[i] = y_old_view[i] / scale_view[i]
                dydt_now_ovr_scale[i] = dydt_old_view[i] / scale_view[i]

            d0 = _norm(y_now_ovr_scale_view)
            d1 = _norm(dydt_now_ovr_scale_view)
            if d0 < 1.e-5 or d1 < 1.e-5:
                h0 = 1.e-6
            else:
                h0 = 0.01 * d0 / d1

            h0_direction = h0 * direction
            t_init_step = t_old + h0_direction
            for i in range(y_size):
                y_init_step[i] = y_old_view[i] + h0_direction * dydt_old_view[i]

            diffeq(t_init_step, y_init_step, dydt_init_step, *args)

            for i in range(y_size):
                dydt_now_ovr_scale[i] = (dydt_init_step_view[i] - dydt_old_view[i]) / scale_view[i]

            d2 = _norm(dydt_now_ovr_scale_view) / h0

            if d1 <= 1.e-15 and d2 <= 1.e-15:
                h1 = max(1.e-6, h0 * 1.e-3)
            else:
                h1 = (0.01 / max(d1, d2))**error_expo

            step_size = min(100. * h0, h1)
    else:
        if first_step <= 0.:
            # Step size must be a positive number
            raise Exception
        elif first_step > t_delta_abs:
            # Step size can not exceed bounds
            raise Exception
        step_size = first_step

    # Initialize RK-K variable
    rk_n_stages_plus1 = rk_n_stages + 1
    K = np.zeros((rk_n_stages_plus1, y_size), dtype=dtype)
    K_tmp = np.empty(y_size, dtype=dtype)
    
    # Setup K memoryviews
    K_view = K
    K_tmp_view = K_tmp

    # Main integration loop
    message = 'Running...'
    # # Time Loop
    while status == 0:

        if t_new == t_end or y_size == 0:
            t_old = t_end
            t_new = t_end
            message = 'Finished'
            status = 1
            break

        # Run RK integration step
        # Determine step size based on previous loop
        min_step = EPS_10
        # Look for over/undershoots in previous step size
        if step_size > max_step:
            step_size = max_step
        elif step_size < min_step:
            step_size = min_step

        # Determine new step size
        step_accepted = False
        step_rejected = False
        step_error = False
        rejected_message = 'Proper step size not found.'

        # # Step Loop
        while not step_accepted:

            if step_size < min_step:
                step_error = True
                rejected_message = 'Required step size is less than spacing between numbers.'
                break

            # Move time forward for this particular step size
            step = step_size * direction
            t_new = t_old + step

            # Check that we are not at the end of integration with that move
            if direction * (t_new - t_end) > 0:
                t_new = t_end

            # Correct the step if we were at the end of integration
            step = t_new - t_old
            step_size = abs(step)

            # Calculate derivative using RK method
            for i in range(y_size):
                K[0, i] = dydt_old_view[i]

            for s in range(1, len_c):

                A_tmp_view = A_view[s, :]
                c = C[s]
                time_ = t_new + c * step
                
                # Dot Product (K, a) * step
                for j in range(s):
                    for i in range(y_size):
                        if j == 0:
                            # Initialize
                            y_tmp[i] = y_old_view[i]

                        y_tmp[i] = y_tmp[i] + (K_view[j, i] * A_tmp_view[j] * step)

                diffeq(time_, y_tmp, K[s, :], *args)
            
            # Dot Product (K, B) * step
            for j in range(rk_n_stages_plus1):                    
                for i in range(y_size):
                    if j == 0:
                        # Initialize
                        y_new[i] = y_old_view[i]
                    
                    y_new[i] = y_new[i] + (K_view[j, i] * B_view[j] * step)

            diffeq(t_new + step, y_new, dydt_new, *args)

            for i in range(y_size):
                # Set last array of the K array.
                K[-1, i] = dydt_new[i]
                
                # Check how well this step performed.
                scale[i] = atol + max(abs(y_old[i]), abs(y_new[i])) * rtol
            
            # Dot Product (K, E) * step / scale
            for j in range(rk_n_stages_plus1):
                for i in range(y_size):
                    if j == 0:
                        # Initialize
                        K_tmp[i] = 0.

                    K_tmp[i] = K_tmp_view[i] + (K_view[j, i] * E_view[j] * step / scale_view[i])

            error_norm = _norm(K_tmp_view)

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
                step_size = step_size * max(MIN_FACTOR, SAFETY * error_norm**-error_expo)
                step_rejected = True

        if not step_accepted or step_error:
            # Issue with step convergence
            status = -1
            message = 'Error in step size calculation:\n' + rejected_message
            break

        # End of step loop. Update the _now variables
        t_old = t_new
        for i in range(y_size):
            y_old[i] = y_new[i]
            dydt_old[i] = dydt_new[i]

        # Save data
        time_domain_list.append(t_new)

        # Numba does not support np.stack(x) if x is a list. So we have to continuously hstack as we go.
        y_results_list.append(y_new.copy())

    y_results = np.stack(y_results_list)
    time_domain = np.asarray(time_domain_list, dtype=np.float64, order='C')
    
    # Set final memory views
    time_domain_view = time_domain
    y_results_view = y_results
    y_results_T_view = y_results_view.T

    # To match the format that scipy follows, we will take the transpose of y.

#     if t_eval.size > 0:
#         # User only wants data at specific points.
#         # The current version of this function has not implemented sicpy's dense output, so we must use an interpolation.
#         t_eval = np.asarray(t_eval, dtype=np.float64)
#         y_results_reduced = np.empty((y_size, t_eval.size), dtype=dtype)

#         for i in range(y_size):
#             # np.interp only works on 1D arrays so we must loop through each of the variables:
#             y_results_reduced[i, :] = np.interp(t_eval, time_domain, y_results[i, :])

#         y_results = y_results_reduced
#         time_domain = t_eval

    if status == 1:
        success = True

    # Make sure arrays are C-contiguous
#     y_results = np.ascontiguousarray(y_results)
#     time_domain = np.ascontiguousarray(time_domain)

    return time_domain, y_results.T, success, message