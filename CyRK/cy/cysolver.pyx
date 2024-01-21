# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.math cimport sqrt, fabs, nextafter, fmax, fmin, isnan, NAN, floor

from cpython.mem cimport PyMem_Free

import numpy as np
cimport numpy as np

from CyRK.utils.utils cimport allocate_mem, reallocate_mem
from CyRK.rk.rk cimport find_rk_properties
from CyRK.cy.common cimport interpolate, SAFETY, MIN_FACTOR, MAX_FACTOR, MAX_STEP, INF, EPS_100, \
    find_expected_size, find_max_num_steps

cdef (double, double) EMPTY_T_SPAN = (NAN, NAN)


cdef class CySolver:
    """
    CySolver: A Object-Oriented Runge-Kutta Solver Implemented in Cython.

    This class provides basic functionality to solve systems of ordinary differential equations using a Runge-Kutta
    scheme. Users can cimport this extension and build their own solvers by overriding its diffeq and update_constants
    methods. Users can also expand on its __init__ or other methods for more complex problems.

    Class attributes are defined, with types, in cysolver.pxd.

    Note: "Time" is used throughout this class's variable names and documentation. It is a placeholder for the
    independent variable that the ODE's derivatives are taken with respect to. Instead of time it could be, for example,
    distance. We choose to use time as a generic placeholder term.

    Attributes
    ----------
    solution_t_view : double[::1]
        Memoryview of the final independent domain found during integration.
        See Also: The public property method, `CySolver.t`
    solution_y_view : double[::1]
        Flattened Memoryview of final solution for dependent variables.
        See Also: The public property method, `CySolver.y`
    solution_extra_view : double[::1]
        Flattened Memoryview of the final solution for any extra parameters captured during integration.
        See Also: The public property method, `CySolver.extra`
    solution_y_ptr : double*
        Pointer of final solution for dependent variables.
    solution_extra_ptr : double*
        Pointer of the final solution for any extra parameters captured during integration.
    solution_t_ptr : double*
        Pointer of the final independent domain found during integration.
    y_size : size_t
        Number of dependent variables in system of ODEs.
    y_size_dbl : double
        Floating point version of y_size.
    y_size_sqrt : double
        Square-root of y_size.
    y0_ptr : double*
        Pointer of dependent variable initial conditions (y0 at t_start).
    t_start : double
        Value of independent variable at beginning of integration (t0).
    t_end : double
        End value of independent variable.
    t_delta : double
        Independent variable domain for integration: t_end - t_start.
        t_delta may be negative or positive depending on if integration is forwards or backwards.
    t_delta_abs : double
        Absolute value of t_delta.
    direction_inf : double
        Direction of integration. If forward then this = +Inf; -Inf otherwise.
    direction_flag : bint
        If True, then integration is in the forward direction.
    num_args : size_t
        Number of additional arguments that the `diffeq` method requires.
    args_ptr : double*
        Pointer of additional arguments used in the `diffeq` method.
    capture_extra bint
        Flag used if extra parameters should be captured during integration.
    num_extra size_t
        Number of extra parameters that should be captured during integration.
    status : char; public
        Numerical flag to indicate status of integrator.
        See "Status and Error Codes.md" in the documentation for more information.
    message : str; public
        Verbal message to accompany `self.status` explaining the state (and potential errors) of the integrator.
    success : bint; public
        Flag indicating if the integration was successful or not.
    rtols_ptr : double*
        Pointer of relative tolerances for each dependent y variable.
    atols_ptr : double*
        Pointer of absolute tolerances for each dependent y variable.
    first_step : double
        Absolute size of the first step to be taken after t_start.
    max_step : double
        Maximum absolute step sized allowed.
    max_num_steps : size_t
        Maximum number of steps allowed before integration auto fails.
    expected_size : size_t
        Anticipated size of integration range, i.e., how many steps will be required.
        Used to build temporary storage arrays for the solution results.
    num_concats : size_t
        Number of concatenations that were required during integration.
        If `expected_size` is too small then it will be expanded as needed. This variable tracks how many expansions
        were required.
        See Also: `CySolver.growths`
    recalc_first_step : bint
        If True, then the `first_step` size is recalculated when `reset_state` is called.
        Flag used when parameters are changed without reinitializing CySolver.
    run_interpolation : bint
        Flag if a final interpolation should be run once integration is completed successfully.
    interpolate_extra : bint
        Flag if interpolation should be run on extra parameters.
        If set to False when `run_interpolation=True`, then interpolation will be run on solution's y, t. These will
        then be used to recalculate extra parameters rather than an interpolation on the extra parameters captured
        during integration.
    len_t_eval : size_t
        Length of user requested independent domain, `t_eval`.
    t_eval_ptr : double*
        Pointer of user requested independent domain, `t_eval`.
    rk : RKConstants
        Instance of a Runge-Kutta Constants class that initializes various RK parameters.
            0: ‘RK23’: Explicit Runge-Kutta method of order 3(2).
            1: ‘RK45’ (default): Explicit Runge-Kutta method of order 5(4).
            2: ‘DOP853’: Explicit Runge-Kutta method of order 8.
    rk_order : size_t
        Runge-Kutta step power.
    error_order : size_t
        Runge-Kutta error power.
    rk_n_stages : size_t
        Number of Runge-Kutta stages performed for each RK step.
    rk_n_stages_plus1 : size_t
        One more than `rk_n_stages`.
    rk_n_stages_extended : size_t
        An extended version of `rk_n_stages` used for DOP853 method.
    error_expo : double
        Exponential used during error calculation. Utilizes `error_order`.
    len_C : size_t
        Size of RK C array.
    len_Arows : size_t
        Number of rows in (unflattened) RK A 2D array.
    len_Acols : size_t
        Number of columns in (unflattened) RK A 2D array.
    A_ptr : double*
        Pointer of (flattened) RK A parameter (data initialized in self.rk instance)
    B_ptr : double*
        Pointer of RK B parameter (data initialized in self.rk instance)
    C_ptr : double*
        Pointer of RK C parameter (data initialized in self.rk instance)
    E_ptr : double*
        Pointer of RK E parameter (data initialized in self.rk instance)
    E3_ptr : double*
        Pointer of RK E3 parameter (data initialized in self.rk instance)
    E5_ptr : double*
        Pointer of RK E5 parameter (data initialized in self.rk instance)
    K_ptr : double*
        Pointer of the RK K parameter.
    t_now : double
        Current value of the independent variable used during integration.
    t_old : double
        Value of the independent variable at the previous step.
    step_size : double
        Current step's absolute size.
    len_t : size_t
        Number of steps taken.
    y_ptr : double*
        Current Pointer of the dependent y variables.
    y_old_ptr : double*
        Pointer of the dependent y variables at the previous step.
    y_ptr : double*
        Current Pointer of dy/dt.
    y_old_ptr : double*
        Pointer of dy/dt at the previous step.
    extra_output_init_ptr : double*
        Pointer of extra outputs at the initial step (t=t0; y=y0).
        Extra outputs are parameters captured during diffeq calculation.
    extra_output_view : double[::1]
        Current Memoryview of extra outputs (at t_now).
        Extra outputs are parameters captured during diffeq calculation.

    Methods
    reset_state()
        Resets the class' state variables so that integration can be rerun.
    calc_first_step()
        Calculates the first step's size.
    rk_step()
        Performs a Runge-Kutta step calculation including local error determination.
    solve(reset=True)
        Public wrapper to the private solve method which calculates the integral of the user-provided system of ODEs.
        If reset=True, `reset_state()` will be called before integration starts.
    _solve()
        Calculates the integral of the user-provided system of ODEs.
        If reset=True, `reset_state()` will be called before integration starts.
    interpolate()
        Performs a final interpolation to fit solution results into a user requested independent variable domain.
    change_t_span(t_span, auto_reset_state=False)
        Public method to change the independent variable limits (start and stop points of integration).
    change_y0(y0, auto_reset_state=False)
        Public method to change the initial conditions.
    change_args(args, auto_reset_state=False)
        Public method to change additional arguments used during integration.
    change_tols(rtol=NAN, atol=NAN, rtols=None, atols=None, auto_reset_state=False)
        Public method to change relative and absolute tolerances and/or their arrays.
    change_max_step(max_step, auto_reset_state=False)
        Public method to change maximum allowed step size.
    change_first_step(first_step, auto_reset_state=False)
        Public method to change first step's size.
    change_t_eval(t_eval, auto_reset_state=False)
        Public method to change user requested independent domain, `t_eval`.
    change_parameters(*, auto_reset_state=True, auto_solve=False)
        Public method to change one or more parameters which have their own `change_*` method.
    update_constants()
        Method that is called during `reset_state` to change any constant parameters used by `diffeq`.
        This method is expected to be overriden by user constructed subclasses.
    diffeq()
        The system of differential equations that will be solved by the integrator.
        This method is expected to be overriden by user constructed subclasses.
    """


    def __init__(self,
            (double, double) t_span,
            const double[::1] y0,
            tuple args = None,
            double rtol = 1.e-3,
            double atol = 1.e-6,
            const double[::1] rtols = None,
            const double[::1] atols = None,
            unsigned char rk_method = 1,
            double max_step = MAX_STEP,
            double first_step = 0.,
            size_t max_num_steps = 0,
            const double[::1] t_eval = None,
            bint capture_extra = False,
            size_t num_extra = 0,
            bint interpolate_extra = False,
            size_t expected_size = 0,
            size_t max_ram_MB = 2000,
            bint call_first_reset = True,
            bint auto_solve = True,
            bint force_fail = False):
        """
        Initialize new CySolver instance.

        Parameters
        ----------
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
        rtols : const double[::1], default=None
            np.ndarray of relative tolerances, one for each dependent y variable.
            None (default) will use the same tolerance (set by `rtol`) for each y variable.
        atols : const double[::1], default=None
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
        call_first_reset : bool, default=True
            If set to True, then the solver will call its `reset_state` method at the end of initialization. This flag
            is overridden by the `auto_solve` flag.
        auto_solve : bint, default=True
            If set to True, then the solver's `solve` method will be called at the end of initialization.
            Otherwise, the user will have to call `solver_instance = CySolver(...); solver_instance.solve()`
            to perform integration.
        """

        # Initialize all class pointers to null
        self.tol_ptrs = NULL
        self.rtols_ptr = NULL
        self.atols_ptr = NULL
        self.solution_y_ptr = NULL
        self.solution_t_ptr = NULL
        self.solution_extra_ptr = NULL
        self.y0_ptr = NULL
        self.args_ptr = NULL
        self.t_eval_ptr = NULL
        self.A_ptr = NULL
        self.B_ptr = NULL
        self.C_ptr = NULL
        self.E_ptr = NULL
        self.E3_ptr = NULL
        self.E5_ptr = NULL
        self.K_ptr = NULL
        self.temporary_y_ptrs = NULL
        self.y_ptr = NULL
        self.y_old_ptr = NULL
        self.dy_ptr = NULL
        self.dy_old_ptr = NULL
        self.extra_output_ptrs = NULL
        self.extra_output_init_ptr = NULL
        self.extra_output_ptr = NULL
        self._solve_time_domain_array_ptr = NULL
        self._solve_y_results_array_ptr = NULL
        self._solve_extra_array_ptr = NULL
        self._interpolate_solution_t_ptr = NULL
        self._interpolate_solution_y_ptr = NULL 
        self._interpolate_solution_extra_ptr = NULL

        # Loop variables
        cdef size_t i

        # Set integration information
        self.status  = -4  # Status code to indicate that integration has not started.
        self.message = 'Integration has not started.'
        self.success = False
        self.recalc_first_step = False
        self.force_fail = force_fail

        # Store y0 values and determine y-size information
        self.y_size = y0.size
        self.y_size_dbl = <double> self.y_size
        self.y_size_sqrt = sqrt(self.y_size_dbl)

        self.y0_ptr = <double *> allocate_mem(self.y_size * sizeof(double), 'y0_ptr (init)')
        for i in range(self.y_size):
            self.y0_ptr[i] = y0[i]

        # Determine time domain information
        self.t_start = t_span[0]
        self.t_end   = t_span[1]
        self.t_delta = self.t_end - self.t_start
        if self.t_delta >= 0.:
            # Integration is moving forward in time.
            self.direction_flag = True
            self.direction_inf  = INF
            self.t_delta_abs    = self.t_delta
        else:
            # Integration is moving backwards in time.
            self.direction_flag = False
            self.direction_inf  = -INF
            self.t_delta_abs    = -self.t_delta

        # Determine integration tolerances
        self.tol_ptrs  = <double *> allocate_mem(2 * self.y_size * sizeof(double), 'tol_ptrs (init)')
        self.rtols_ptr = &self.tol_ptrs[0]
        self.atols_ptr = &self.tol_ptrs[self.y_size]
        cdef double rtol_tmp, rtol_min
        rtol_min = INF

        if rtols is not None:
            # User provided an arrayed version of rtol.
            if len(rtols) != self.y_size:
                raise AttributeError('rtol array must be the same size as y0.')
            for i in range(self.y_size):
                rtol_tmp = rtols[i]
                # Check that the tolerances are not too small.
                if rtol_tmp < EPS_100:
                    rtol_tmp = EPS_100
                rtol_min = fmin(rtol_min, rtol_tmp)
                self.rtols_ptr[i] = rtol_tmp
        else:
            # No array provided. Use the same rtol for all ys.
            # Check that the tolerances are not too small.
            if rtol < EPS_100:
                rtol = EPS_100
            rtol_min = rtol
            for i in range(self.y_size):
                self.rtols_ptr[i] = rtol

        if atols is not None:
            # User provided an arrayed version of atol.
            if len(atols) != self.y_size:
                raise AttributeError('atol array must be the same size as y0.')
            for i in range(self.y_size):
                self.atols_ptr[i] = atols[i]
        else:
            # No array provided. Use the same atol for all ys.
            for i in range(self.y_size):
                self.atols_ptr[i] = atol
        
        # Determine extra outputs
        self.capture_extra = capture_extra
        # To avoid memory access violations we need to set the extra output arrays no matter if they are used.
        # If not used, just set them to size zero.
        if self.capture_extra:
            if num_extra <= 0:
                self.status = -8
                raise AttributeError('Capture extra set to True, but number of extra set to 0 (or negative).')
            self.num_extra = num_extra
        else:
            # Even though we are not capturing extra, we still want num_extra to be equal to 1 so that nan arrays
            # are properly initialized
            self.num_extra = 1

        # Expected size of output arrays.
        if expected_size == 0:
            # CySolver will attempt to guess on a best size for the arrays.
            self.expected_size = find_expected_size(
                self.y_size,
                num_extra,
                self.t_delta_abs,
                rtol_min,
                capture_extra,
                False)
        else:
            self.expected_size = expected_size
        # Set the current size to the expected size.
        # `expected_size` should never change but current might grow if expected size is not large enough.
        self.current_size = self.expected_size

        # Determine max number of steps
        find_max_num_steps(
            self.y_size,
            self.num_extra,
            max_num_steps,
            max_ram_MB,
            capture_extra,
            False,
            &self.user_provided_max_num_steps,
            &self.max_num_steps)

        # This variable tracks how many times the storage arrays have been appended.
        # It starts at 1 since there is at least one storage array present.
        self.num_concats = 1

        # Determine optional arguments
        if args is None:
            self.use_args = False
            # Even though there are no args set arg size to 1 to initialize nan arrays
            self.num_args = 1
        else:
            self.use_args = True
            self.num_args = len(args)
        self.args_ptr = <double *> allocate_mem(self.num_args * sizeof(double), 'args_ptr (init)')
        for i in range(self.num_args):
            if self.use_args:
                self.args_ptr[i] = args[i]
            else:
                self.args_ptr[i] = NAN

        # Initialize live variable arrays
        self.temporary_y_ptrs = <double *> allocate_mem(4 * self.y_size * sizeof(double), 'temporary_y_ptrs (init)')
        self.y_ptr      = &self.temporary_y_ptrs[0 * self.y_size]
        self.y_old_ptr  = &self.temporary_y_ptrs[1 * self.y_size]
        self.dy_ptr     = &self.temporary_y_ptrs[2 * self.y_size]
        self.dy_old_ptr = &self.temporary_y_ptrs[3 * self.y_size]

        self.extra_output_ptrs = <double *> allocate_mem(2 * self.num_extra * sizeof(double), 'extra_output_ptrs (init)')

        self.extra_output_init_ptr = &self.extra_output_ptrs[0 * self.num_extra]
        self.extra_output_ptr      = &self.extra_output_ptrs[1 * self.num_extra]
       
        for i in range(self.num_extra):
            self.extra_output_init_ptr[i] = NAN
            self.extra_output_ptr[i]      = NAN

        # Initialize storage variables
        self.solution_t_ptr = <double *> allocate_mem(sizeof(double), 'solution_t_ptr (init)')
        self.solution_t_ptr[0] = NAN

        self.solution_y_ptr = <double *> allocate_mem(self.y_size * sizeof(double), 'solution_y_ptr (init)')
        for i in range(self.y_size):
            self.solution_y_ptr[i] = NAN

        self.solution_extra_ptr = <double *> allocate_mem(self.num_extra * sizeof(double), 'solution_extra_ptr (init)')
        for i in range(self.num_extra):
            self.solution_extra_ptr[i] = NAN

        # Have solution memoryviews point to these addresses
        self.solution_t_view     = <double[:1]> self.solution_t_ptr
        self.solution_y_view     = <double[:self.y_size]> self.solution_y_ptr
        self.solution_extra_view = <double[:self.num_extra]> self.solution_extra_ptr

        # Determine interpolation information
        if t_eval is None:
            self.run_interpolation = False
            self.interpolate_extra = False
            # Even though we are not using t_eval, set its size equal to one so that nan arrays can be built
            self.len_t_eval = 1
        else:
            self.run_interpolation = True
            self.interpolate_extra = interpolate_extra
            self.len_t_eval = len(t_eval)

        self.t_eval_ptr = <double *> allocate_mem(self.len_t_eval * sizeof(double), 't_eval_ptr (init)')
        for i in range(self.len_t_eval):
            if self.run_interpolation:
                self.t_eval_ptr[i] = t_eval[i]
            else:
                self.t_eval_ptr[i] = NAN

        # Initialize RK arrays
        self.rk_method = rk_method
        find_rk_properties(
            self.rk_method,
            &self.rk_order,
            &self.error_order,
            &self.rk_n_stages,
            &self.len_Arows,
            &self.len_Acols,
            &self.A_ptr,
            &self.B_ptr,
            &self.C_ptr,
            &self.E_ptr,
            &self.E3_ptr,
            &self.E5_ptr
            )

        if self.rk_order == 0:
            raise AttributeError('Unknown or not-yet-implemented RK method requested.')
        
        self.len_C             = self.rk_n_stages
        self.rk_n_stages_plus1 = self.rk_n_stages + 1
        self.error_expo        = 1. / (<double>self.error_order + 1.)

        # Initialize other RK-related Arrays
        self.K_ptr = <double *> allocate_mem(self.rk_n_stages_plus1 * self.y_size * sizeof(double), 'K_ptr (init)')

        # Store user provided step information
        self.first_step = first_step
        self.max_step   = max_step

        # Parameters are initialized but may not be set to correct values.
        # Call reset state to ensure everything is ready.
        if call_first_reset or auto_solve:
            self.reset_state()

        # Run solver if requested
        if auto_solve:
            # We know for a fact that this is the first time solve will be called and we just reset the Sovler's state
            # So we can safely tell the solve method not to reset.
            self._solve(reset=False)

    cpdef void reset_state(self):
        """ Resets the class' state variables so that integration can be rerun. """
        cdef size_t i, j
        cdef double temp_double

        # Set current and old time variables equal to t0
        self.t_old = self.t_start
        self.t_now = self.t_start
        # Track number of steps.
        # Initial conditions were provided so the number of steps is already 1
        self.len_t = 1

        # It is important K be initialized with 0s
        for i in range(self.rk_n_stages_plus1):
            for j in range(self.y_size):
                self.K_ptr[i * self.y_size + j] = 0.

                # While we have this loop; set y back to initial conditions
                if i == 0:
                    temp_double       = self.y0_ptr[j]
                    self.y_ptr[j]     = temp_double
                    self.y_old_ptr[j] = temp_double

        # Update any constant parameters that the user has set
        self.update_constants()

        # Make initial call to diffeq()
        self.diffeq()

        # Store first dydt
        for i in range(self.y_size):
            self.dy_old_ptr[i] = self.dy_ptr[i]

        # Store extra outputs for the first time step
        if self.capture_extra:
            for i in range(self.num_extra):
                self.extra_output_init_ptr[i] = self.extra_output_ptr[i]

        # Determine first step's size
        if self.first_step == 0. or self.recalc_first_step:
            self.step_size = self.calc_first_step()
        else:
            if self.first_step <= 0.:
                self.status = -8
                self.message = "Attribute error."
                raise AttributeError('Error in user-provided step size: Step size must be a positive number.')
            elif self.first_step > self.t_delta_abs:
                self.status = -8
                self.message = "Attribute error."
                raise AttributeError('Error in user-provided step size: Step size can not exceed bounds.')
            self.step_size = self.first_step

        # Reset output storage
        self.current_size = self.expected_size
        self.num_concats = 1

        # Reset storage variables to clear any old solutions and to avoid access violations if solve() is not called.
        self.solution_t_ptr[0] = NAN
        for i in range(self.y_size):
            self.solution_y_ptr[i] = NAN
        for i in range(self.num_extra):
            self.solution_extra_ptr[i] = NAN

        # Have solution memoryviews point to these addresses
        self.solution_t_view     = <double[:1]> self.solution_t_ptr
        self.solution_y_view     = <double[:self.y_size]> self.solution_y_ptr
        self.solution_extra_view = <double[:self.num_extra]> self.solution_extra_ptr

        # Other integration flags and messages
        self.success = False
        self.status = -5  # status == -5 means that reset has been called but solve has not yet been called.
        self.message = "CySolver has been reset."

    cdef double calc_first_step(self) noexcept nogil:
        """
        Select an initial step size based on the differential equation.
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4.
        """

        cdef double step_size, d0, d1, d2, d0_abs, d1_abs, d2_abs, h0, h1, scale
        cdef double y_old_tmp

        if self.y_size == 0:
            step_size = INF
        else:
            # Find the norm for d0 and d1
            d0 = 0.
            d1 = 0.
            for i in range(self.y_size):
                y_old_tmp = self.y_old_ptr[i]

                scale = self.atols_ptr[i] + fabs(y_old_tmp) * self.rtols_ptr[i]

                d0_abs = fabs(y_old_tmp / scale)
                d1_abs = fabs(self.dy_old_ptr[i] / scale)
                d0 += (d0_abs * d0_abs)
                d1 += (d1_abs * d1_abs)

            d0 = sqrt(d0) / self.y_size_sqrt
            d1 = sqrt(d1) / self.y_size_sqrt

            if d0 < 1.e-5 or d1 < 1.e-5:
                h0 = 1.e-6
            else:
                h0 = 0.01 * d0 / d1

            if self.direction_flag:
                h0_direction = h0
            else:
                h0_direction = -h0

            self.t_now = self.t_old + h0_direction
            for i in range(self.y_size):
                self.y_ptr[i] = self.y_old_ptr[i] + h0_direction * self.dy_old_ptr[i]

            # Update dy_new_view
            self.diffeq()

            # Find the norm for d2
            d2 = 0.
            for i in range(self.y_size):

                scale = self.atols_ptr[i] + fabs(self.y_old_ptr[i]) * self.rtols_ptr[i]
                d2_abs = fabs( (self.dy_ptr[i] - self.dy_old_ptr[i]) / scale)
                d2 += (d2_abs * d2_abs)

            d2 = sqrt(d2) / (h0 * self.y_size_sqrt)

            if d1 <= 1.e-15 and d2 <= 1.e-15:
                h1 = fmax(1.e-6, h0 * 1.e-3)
            else:
                h1 = (0.01 / fmax(d1, d2))**self.error_expo

            step_size = fmax(10. * fabs(nextafter(self.t_old, self.direction_inf) - self.t_old),
                            fmin(100. * h0, h1))

        return step_size

    cpdef void solve(
            self,
            bint reset = True
            ):
        """
        Public wrapper to the private solve method which calculates the integral of the user-provided system of ODEs.
        
        Parameters
        ----------
        reset : bint, default=True
            If True, `reset_state()` will be called before integration starts.
        """
        self._solve(reset=reset)

    cdef void _solve(
            self,
            bint reset = True
            ):
        """
        Calculates the integral of the user-provided system of ODEs.
        
        Parameters
        ----------
        reset : bint, default=True
            If True, `reset_state()` will be called before integration starts.
        """

        # Reset the solver's state (avoid issues if solve() is called multiple times).
        if reset:
            self.reset_state()

        # Setup loop variables
        cdef size_t i
        cdef int rk_step_output

        # Setup storage arrays
        # These arrays are built to fit a number of points equal to `self.expected_size`
        # If the integration needs more than that then a new array will be concatenated (with performance costs) to these.        
        if self._solve_time_domain_array_ptr is NULL:
            self._solve_time_domain_array_ptr = <double *> allocate_mem(
                self.current_size * sizeof(double),
                '_solve_time_domain_array_ptr (_solve)')
        else:
            self._solve_time_domain_array_ptr = <double *> reallocate_mem(
                self._solve_time_domain_array_ptr,
                self.current_size * sizeof(double),
                '_solve_time_domain_array_ptr (_solve)')

        if self._solve_y_results_array_ptr is NULL:
            self._solve_y_results_array_ptr = <double *> allocate_mem(
                self.y_size * self.current_size * sizeof(double),
                '_solve_y_results_array_ptr (_solve)')
        else:
            self._solve_y_results_array_ptr = <double *> reallocate_mem(
                self._solve_y_results_array_ptr,
                self.y_size * self.current_size * sizeof(double),
                '_solve_y_results_array_ptr (_solve)')
        
        if self.capture_extra:
            if self._solve_extra_array_ptr is NULL:
                self._solve_extra_array_ptr = <double *> allocate_mem(
                    self.num_extra * self.current_size * sizeof(double),
                    '_solve_extra_array_ptr (_solve)')
            else:
                self._solve_extra_array_ptr = <double *> reallocate_mem(
                    self._solve_extra_array_ptr,
                    self.num_extra * self.current_size * sizeof(double),
                    '_solve_extra_array_ptr (_solve)')

        # Load initial conditions into storage arrays
        self._solve_time_domain_array_ptr[0] = self.t_start
        for i in range(self.y_size):
            self._solve_y_results_array_ptr[i] = self.y0_ptr[i]
        if self.capture_extra:
            for i in range(self.num_extra):
                self._solve_extra_array_ptr[i] = self.extra_output_init_ptr[i]

        # # Main integration loop
        self.status = 0

        if self.y_size == 0:
            self.status = -6

        while self.status == 0:

            # Check that integration is not complete.
            if self.t_now == self.t_end:
                self.t_old = self.t_end
                self.status = 1
                break

            # Check that maximum number of steps has not been exceeded.
            if self.len_t > self.max_num_steps:
                if self.user_provided_max_num_steps:
                    self.status = -2
                else:
                    self.status = -3
                break

            # # Perform RK Step
            rk_step_output = rk_step_cf(
                self.diffeq,
                self,
                self.t_end,
                self.direction_flag,
                self.direction_inf,
                self.y_size,
                self.y_size_dbl,
                self.y_size_sqrt,
                &self.t_now,
                self.y_ptr,
                self.dy_ptr,
                &self.t_old,
                self.y_old_ptr,
                self.dy_old_ptr,
                &self.step_size,
                &self.status,
                self.atols_ptr,
                self.rtols_ptr,
                self.max_step,
                self.rk_method,
                self.rk_n_stages,
                self.rk_n_stages_plus1,
                self.len_Acols,
                self.len_C,
                self.A_ptr,
                self.B_ptr,
                self.C_ptr,
                self.K_ptr,
                self.E_ptr,
                self.E3_ptr,
                self.E5_ptr,
                self.error_expo,
                MIN_FACTOR,
                MAX_FACTOR,
                SAFETY
                )

            # Check if an error occurred during step calculations before storing data.
            if self.status != 0:
                break

            # Store data
            if self.len_t >= self.current_size:
                # There is more data then we have room in our arrays.
                # Build new arrays with more space.
                # OPT: Note this is an expensive operation.
                self.num_concats += 1

                # Grow the array by 50% its current value
                self.current_size = <size_t> floor(<double>self.current_size * (1.5))

                self._solve_time_domain_array_ptr = <double *> reallocate_mem(
                    self._solve_time_domain_array_ptr,
                    self.current_size * sizeof(double),
                    'self._solve_time_domain_array_ptr (_solve; growth stage)')

                self._solve_y_results_array_ptr = <double *> reallocate_mem(
                    self._solve_y_results_array_ptr,
                    self.y_size * self.current_size * sizeof(double),
                    'self._solve_y_results_array_ptr (_solve; growth stage)')

                if self.capture_extra:
                    self._solve_extra_array_ptr = <double *> reallocate_mem(
                        self._solve_extra_array_ptr,
                        self.num_extra * self.current_size * sizeof(double),
                        'self._solve_extra_array_ptr (_solve; growth stage)')

            # Add this step's results to our storage arrays.
            self._solve_time_domain_array_ptr[self.len_t] = self.t_now
            for i in range(self.y_size):
                self._solve_y_results_array_ptr[self.len_t * self.y_size + i] = self.y_ptr[i]

            if self.capture_extra:
                for i in range(self.num_extra):
                    self._solve_extra_array_ptr[self.len_t * self.num_extra + i] = self.extra_output_ptr[i]

            # Increase number of independent variable points.
            self.len_t += 1

        # Integration has stopped. Check if it was successful.
        if self.status != 1 or self.force_fail:
            self.success = False
        else:
            self.success = True

        if self.success:
            # Solution was successful.
            # Free any data stored in solution arrays so that we can repopulate them with the new solution
            if not (self.solution_t_ptr is NULL):
                PyMem_Free(self.solution_t_ptr)
                self.solution_t_ptr = NULL
            if not (self.solution_y_ptr is NULL):
                PyMem_Free(self.solution_y_ptr)
                self.solution_y_ptr = NULL
            if self.capture_extra:
                if not (self.solution_extra_ptr is NULL):
                    PyMem_Free(self.solution_extra_ptr)
                    self.solution_extra_ptr = NULL

            # Load values into solution arrays.
            # The arrays built during integration likely have a bunch of unused junk at the end due to overbuilding their size.
            # This process will remove that junk and leave only the valid data.
            # These arrays will always be the same length or less (self.len_t <= self.current_size) than the ones they are
            # built off of, so it is safe to use Realloc.
            self.solution_t_ptr = <double *> reallocate_mem(
                self._solve_time_domain_array_ptr,
                self.len_t * sizeof(double),
                'solution_t_ptr (_solve; success stage)')
            self._solve_time_domain_array_ptr = NULL

            self.solution_y_ptr = <double *> reallocate_mem(
                self._solve_y_results_array_ptr,
                self.y_size * self.len_t * sizeof(double),
                'solution_y_ptr (_solve; success stage)')
            self._solve_y_results_array_ptr = NULL

            if self.capture_extra:
                self.solution_extra_ptr = <double *> reallocate_mem(
                    self._solve_extra_array_ptr,
                    self.num_extra * self.len_t * sizeof(double),
                    'solution_extra_ptr (_solve; success stage)')
                self._solve_extra_array_ptr = NULL
        else:
            # Integration was not successful. Make solution pointers length 1 nan arrays.
            # Clear the storage arrays used during the step loop
            if not (self._solve_time_domain_array_ptr is NULL):
                PyMem_Free(self._solve_time_domain_array_ptr)
                self._solve_time_domain_array_ptr = NULL
            if not (self._solve_y_results_array_ptr is NULL):
                PyMem_Free(self._solve_y_results_array_ptr)
                self._solve_y_results_array_ptr = NULL
            if self.capture_extra:
                if not (self._solve_extra_array_ptr is NULL):
                    PyMem_Free(self._solve_extra_array_ptr)
                    self._solve_extra_array_ptr = NULL

            # We still need to build solution arrays so that accessing the solution will not cause access violations.
            # Build size-1 arrays. Since the solution was not successful, set all arrays to NANs
            self.solution_t_ptr = <double *> reallocate_mem(
                self.solution_t_ptr,
                sizeof(double),
                'solution_t_ptr (_solve; fail stage)')
            self.solution_t_ptr[0] = NAN

            self.solution_y_ptr = <double *> reallocate_mem(
                self.solution_y_ptr,
                self.y_size * sizeof(double),
                'solution_y_ptr (_solve; fail stage)')
            for i in range(self.y_size):
                self.solution_y_ptr[i] = NAN

            self.solution_extra_ptr = <double *> reallocate_mem(
                self.solution_extra_ptr,
                self.num_extra * sizeof(double),
                'solution_extra_ptr (_solve; fail stage)')
            for i in range(self.num_extra):
                self.solution_extra_ptr[i] = NAN

        # Integration is complete. Check if interpolation was requested.
        if self.success:
            if self.run_interpolation:
                self.interpolate()
                # Use different len_t
                self.len_t_touse = self.len_t_eval
            else:
                self.len_t_touse = self.len_t
        else:
            # If integration was not successful use t_len = 1 to allow for nan arrays
            self.len_t_touse = 1

        # Convert solution pointers to a more user-friendly memoryview format.
        # Define post-run variables
        cdef size_t y_size_touse, extra_size_touse
        y_size_touse     = self.y_size * self.len_t_touse
        extra_size_touse = self.num_extra * self.len_t_touse

        self.solution_t_view     = <double[:self.len_t_touse]> self.solution_t_ptr
        self.solution_y_view     = <double[:y_size_touse]> self.solution_y_ptr
        self.solution_extra_view = <double[:extra_size_touse]> self.solution_extra_ptr

        # Update integration message
        if self.status == 1:
            self.message = "Integration completed without issue."
        elif self.status == 0:
            self.message = "Integration is/was ongoing (perhaps it was interrupted?)."
        elif self.status == -1:
            self.message = "Error in step size calculation:\n\tRequired step size is less than spacing between numbers."
        elif self.status == -2:
            self.message = "Maximum number of steps (set by user) exceeded during integration."
        elif self.status == -3:
            self.message = "Maximum number of steps (set by system architecture) exceeded during integration."
        elif self.status == -6:
            self.message = "Integration never started: y-size is zero."
        elif self.status == -7:
            self.message = "Error in step size calculation:\n\tError in step size acceptance."

        # Release any memory that may still be alive due to exceptions being raised.
        if not (self._solve_time_domain_array_ptr is NULL):
            PyMem_Free(self._solve_time_domain_array_ptr)
            self._solve_time_domain_array_ptr = NULL
        if not (self._solve_y_results_array_ptr is NULL):
            PyMem_Free(self._solve_y_results_array_ptr)
            self._solve_y_results_array_ptr = NULL
        if not (self._solve_extra_array_ptr is NULL):
            PyMem_Free(self._solve_extra_array_ptr)
            self._solve_extra_array_ptr = NULL

    cdef void interpolate(self):
        """ Interpolate the results of a successful integration over the user provided time domain, `t_eval`. """
        # User only wants data at specific points.
        cdef char old_status
        old_status = self.status
        self.status = 2  # Interpolation is being performed.

        # Setup loop variables
        cdef size_t i, j

        # Check to make sure that t-eval is set
        if not self.t_eval_ptr:
            raise ValueError('Interpolation function called but t_eval_ptr is null.')

        # TODO: The current version of CySolver has not implemented sicpy's dense output. Instead we use an interpolation.
        # Build final interpolated time and solution arrays
        if self._interpolate_solution_t_ptr is NULL:
            self._interpolate_solution_t_ptr = <double *> allocate_mem(
                self.len_t_eval * sizeof(double),
                '_interpolate_solution_t_ptr (interpolate)')
        else:
            self._interpolate_solution_t_ptr = <double *> reallocate_mem(
                self._interpolate_solution_t_ptr,
                self.len_t_eval * sizeof(double),
                '_interpolate_solution_t_ptr (interpolate)')

        if self._interpolate_solution_y_ptr is NULL:
            self._interpolate_solution_y_ptr = <double *> allocate_mem(
                self.y_size * self.len_t_eval * sizeof(double),
                'self._interpolate_solution_y_ptr (interpolate)')
        else:
            self._interpolate_solution_y_ptr = <double *> reallocate_mem(
                self._interpolate_solution_y_ptr,
                self.y_size * self.len_t_eval * sizeof(double),
                'self._interpolate_solution_y_ptr (interpolate)')

        # Perform interpolation on y values
        interpolate(self.solution_t_ptr, self.t_eval_ptr, self.solution_y_ptr, self._interpolate_solution_y_ptr,
                    self.len_t, self.len_t_eval, self.y_size, False)

        # Make a copy of t_eval (issues can arise if we store the t_eval pointer in solution array).
        for i in range(self.len_t_eval):
            self._interpolate_solution_t_ptr[i] = self.t_eval_ptr[i]

        if self.capture_extra:
            # Right now if there is any extra output then it is stored at each time step used in the RK loop.
            # We have to make a choice:
            #   - Do we interpolate the extra values that were stored?
            #   - Or do we use the interpolated t, y values to find new extra parameters at those specific points.
            # The latter method is more computationally expensive (recalls the diffeq for each y) but is more accurate.
            # This decision is set by the user with the `interpolate_extra` flag.

            # Build final interpolated solution array (Used if self.interpolate_extra is True or False)
            if self._interpolate_solution_extra_ptr is NULL:
                self._interpolate_solution_extra_ptr = <double *> allocate_mem(
                    self.num_extra * self.len_t_eval * sizeof(double),
                    'self._interpolate_solution_extra_ptr (interpolate)')
            else:
                self._interpolate_solution_extra_ptr = <double *> reallocate_mem(
                    self._interpolate_solution_extra_ptr,
                    self.num_extra * self.len_t_eval * sizeof(double),
                    'self._interpolate_solution_extra_ptr (interpolate)')

            if self.interpolate_extra:
                # Perform interpolation on extra outputs
                interpolate(
                    self.solution_t_ptr, self.t_eval_ptr, self.solution_extra_ptr, self._interpolate_solution_extra_ptr,
                    self.len_t, self.len_t_eval, self.num_extra, False)
            else:
                # Use the new interpolated y and t values to recalculate the extra outputs with self.diffeq
                for i in range(self.len_t_eval):
                    # Set state variables
                    self.t_now = self.t_eval_ptr[i]
                    for j in range(self.y_size):
                        self.y_ptr[j] = self._interpolate_solution_y_ptr[i * self.y_size + j]

                    # Call diffeq to recalculate extra outputs
                    self.diffeq()

                    # Capture extras
                    for j in range(self.num_extra):
                        self._interpolate_solution_extra_ptr[i * self.num_extra + j] = self.extra_output_ptr[j]

            # Replace old pointers with new interpolated pointers and release the memory for the old stuff
            if not (self.solution_extra_ptr is NULL):
                PyMem_Free(self.solution_extra_ptr)
            self.solution_extra_ptr = self._interpolate_solution_extra_ptr
            self._interpolate_solution_extra_ptr = NULL

        # Replace old pointers with new interpolated pointers and release the memory for the old stuff
        if not (self.solution_t_ptr is NULL):
            PyMem_Free(self.solution_t_ptr)
        self.solution_t_ptr = self._interpolate_solution_t_ptr
        self._interpolate_solution_t_ptr = NULL
        if not (self.solution_y_ptr is NULL):
            PyMem_Free(self.solution_y_ptr)
        self.solution_y_ptr = self._interpolate_solution_y_ptr
        self._interpolate_solution_y_ptr = NULL

        # Interpolation is done.
        self.status = old_status

        # Free any memory that may still be alive if exceptions were raised.
        if not (self._interpolate_solution_t_ptr is NULL):
            PyMem_Free(self._interpolate_solution_t_ptr)
            self._interpolate_solution_t_ptr = NULL
        if not (self._interpolate_solution_y_ptr is NULL):
            PyMem_Free(self._interpolate_solution_y_ptr)
            self._interpolate_solution_y_ptr = NULL
        if not (self._interpolate_solution_extra_ptr is NULL):
            PyMem_Free(self._interpolate_solution_extra_ptr)
            self._interpolate_solution_extra_ptr = NULL

    cpdef void change_t_span(
            self,
            (double, double) t_span,
            bint auto_reset_state = False
            ):
        """
        Public method to change the independent variable limits (start and stop points of integration).
        
        Parameters
        ----------
        t_span : (double, double)
            New t_span to use during integration.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # Update time domain information
        self.t_start     = t_span[0]
        self.t_end       = t_span[1]
        self.t_delta     = self.t_end - self.t_start
        if self.t_delta >= 0.:
            self.direction_flag = True
            self.direction_inf  = INF
            self.t_delta_abs    = self.t_delta
        else:
            self.direction_flag = False
            self.direction_inf  = -INF
            self.t_delta_abs    = -self.t_delta

        # A change to t-span will affect the first step's size
        self.recalc_first_step = True

        if auto_reset_state:
            self.reset_state()

    cpdef void change_y0(
            self,
            const double[::1] y0,
            bint auto_reset_state = False
            ):
        """
        Public method to change the initial conditions.
        
        Note: the size of y0 can not be different from the original y0 used to instantiate the class instance.

        Parameters
        ----------
        y0 : double[::1]
            New dependent variable initial conditions.
            Must be the same size as the original y0.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # Check y-size information
        cdef size_t i, y_size_new
        y_size_new = len(y0)

        if self.y_size != y_size_new:
            # So many things need to update if ysize changes that the user should just create a new class instance.
            self.status = -8
            self.message = "Attribute error."
            raise AttributeError('New y0 must be the same size as the original y0 used to create CySolver class.'
                                 'Create new CySolver instance instead.')

        # Store y0 values for later
        for i in range(self.y_size):
            self.y0_ptr[i] = y0[i]

        # A change to y0 will affect the first step's size
        self.recalc_first_step = True

        if auto_reset_state:
            self.reset_state()

    cdef void change_y0_pointer(
                self,
                double* y0_ptr,
                bint auto_reset_state = False
                ):
            """
            Public method to change the initial conditions.

            Note: the size of y0 can not be different from the original y0 used to instantiate the class instance.

            Parameters
            ----------
            y0 : double*
                New pointer to dependent variable initial conditions.
                Must be the same size as the original y0.
            auto_reset_state : bint, default=False
                If True, then the `reset_state` method will be called once parameter is changed.
            """

            # This function is not as safe as `change_y0` as it assumes that the user provided the same length y0.

            # Check y-size information
            cdef size_t i

            # Store y0 values for later
            for i in range(self.y_size):
                self.y0_ptr[i] = y0_ptr[i]

            # A change to y0 will affect the first step's size
            self.recalc_first_step = True

            if auto_reset_state:
                self.reset_state()

    cpdef void change_args(
            self,
            tuple args,
            bint auto_reset_state = False
            ):
        """
        Public method to change additional arguments used during integration.

        Parameters
        ----------
        args : tuple
            New tuple of additional arguments.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # Determine optional arguments
        if args is None:
            self.use_args = False
            # Even though there are no args set arg size to 1 to initialize nan arrays
            self.num_args = 1
        else:
            self.use_args = True
            self.num_args = len(args)

        if self.args_ptr is NULL:
            self.args_ptr = <double *> allocate_mem(
                self.num_args * sizeof(double),
                'args_ptr (change_args)')
        else:
            self.args_ptr = <double *> reallocate_mem(
                self.args_ptr,
                self.num_args * sizeof(double),
                'args_ptr (change_args)')

        for i in range(self.num_args):
            if self.use_args:
                self.args_ptr[i] = args[i]
            else:
                self.args_ptr[i] = NAN

        # A change to args will affect the first step's size
        self.recalc_first_step = True

        if auto_reset_state:
            self.reset_state()

    cpdef void change_tols(
            self,
            double rtol = NAN,
            double atol = NAN,
            const double[::1] rtols = None,
            const double[::1] atols = None,
            bint auto_reset_state = False
            ):
        """
        Public method to change relative and absolute tolerances and/or their arrays.
        
        Parameters
        ----------
        rtol : double, default=NAN
            New relative tolerance for all dependent y variables.
            if NAN (the default), then no change will be made.
        atol : double, default=NAN
            New absolute tolerance for all dependent y variables.
            if NAN (the default), then no change will be made.
        rtols : const double[::1]
            Numpy ndarray of relative tolerances, one for each dependent y variable.
            if None (the default), then no change will be made.
        atols : const double[::1]
            Numpy ndarray of absolute tolerances, one for each dependent y variable.
            if None (the default), then no change will be made.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # This is one of the few change functions where nothing might change.
        # Track if updates need to be made
        cdef bint something_changed = False

        # Update tolerances
        cdef double rtol_tmp

        if rtols is not None or not isnan(rtol):
            # Change to rtol
            something_changed = True

            if rtols is not None:
                # Using arrayed rtol
                if len(rtols) != self.y_size:
                    raise AttributeError('rtols must be the same size as y0.')
                for i in range(self.y_size):
                    rtol_tmp = rtols[i]
                    if rtol_tmp < EPS_100:
                        rtol_tmp = EPS_100
                    self.rtols_ptr[i] = rtol_tmp
            elif not isnan(rtol):
                # Using constant rtol
                # Check tolerances
                if rtol < EPS_100:
                    rtol = EPS_100
                for i in range(self.y_size):
                    self.rtols_ptr[i] = rtol

        if atols is not None or not isnan(atol):
            # Change to atol
            something_changed = True

            if atols is not None:
                # Using arrayed atol
                if len(atols) != self.y_size:
                    raise AttributeError('atols must be the same size as y0.')
                for i in range(self.y_size):
                    self.atols_ptr[i] = atols[i]
            elif not isnan(atol):
                for i in range(self.y_size):
                    self.atols_ptr[i] = atol

        if something_changed:
            # A change to tolerances will affect the first step's size
            self.recalc_first_step = True

            if auto_reset_state:
                self.reset_state()

    cpdef void change_max_step(
            self,
            double max_step,
            bint auto_reset_state = False
            ):
        """
        Public method to change maximum allowed step size.
        
        Parameters
        ----------
        max_step : double
            New maximum step size used during integration.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        self.max_step = fabs(max_step)

        if auto_reset_state:
            self.reset_state()

    cpdef void change_first_step(
            self,
            double first_step,
            bint auto_reset_state = False
            ):
        """
        Public method to change first step's size.
        
        Parameters
        ----------
        first_step : double
            New first step's size.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        self.first_step = first_step
        if self.first_step == 0.:
            self.step_size = self.calc_first_step()
        else:
            if self.first_step <= 0.:
                self.status = -8
                self.message = "Attribute error."
                raise AttributeError('Error in user-provided step size: Step size must be a positive number.')
            elif self.first_step > self.t_delta_abs:
                self.status = -8
                self.message = "Attribute error."
                raise AttributeError('Error in user-provided step size: Step size can not exceed bounds.')
            self.step_size = self.first_step

        # If first step has already been reset then no need to call it again later.
        self.recalc_first_step = False

        if auto_reset_state:
            self.reset_state()

    cpdef void change_t_eval(
            self,
            const double[::1] t_eval,
            bint auto_reset_state = False
            ):
        """
        Public method to change user requested independent domain, `t_eval`.

        Parameters
        ----------
        t_eval : double[:]
            New independent domain at which solution will be interpolated.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        cdef size_t i

        # Determine interpolation information
        self.run_interpolation = True
        self.len_t_eval = len(t_eval)

        if self.t_eval_ptr is NULL:
            self.t_eval_ptr = <double *> allocate_mem(
                self.len_t_eval * sizeof(double),
                't_eval_ptr (change_t_eval)')
        else:
            self.t_eval_ptr = <double *> reallocate_mem(
                self.t_eval_ptr,
                self.len_t_eval * sizeof(double),
                't_eval_ptr (change_t_eval)')

        if self.run_interpolation:
            for i in range(self.len_t_eval):
                self.t_eval_ptr[i] = t_eval[i]

        if auto_reset_state:
            self.reset_state()

    cdef void change_t_eval_pointer(
            self,
            double* new_t_eval_ptr,
            size_t new_len_t_eval,
            bint auto_reset_state = False
            ):
        """
        Public method to change user requested independent domain, `t_eval`.

        Parameters
        ----------
        t_eval_ptr : double[:]
            New pointer to independent domain at which solution will be interpolated.
        auto_reset_state : bint, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # Determine interpolation information
        self.run_interpolation = True
        self.len_t_eval = new_len_t_eval

        if self.t_eval_ptr is NULL:
            self.t_eval_ptr = <double *> allocate_mem(
                self.len_t_eval * sizeof(double),
                't_eval_ptr (change_t_eval_pointer)')
        else:
            self.t_eval_ptr = <double *> reallocate_mem(
                self.t_eval_ptr,
                self.len_t_eval * sizeof(double),
                't_eval_ptr (change_t_eval_pointer)')

        if self.run_interpolation:
            for i in range(self.len_t_eval):
                self.t_eval_ptr[i] = new_t_eval_ptr[i]

        if auto_reset_state:
            self.reset_state()

    cpdef void change_parameters(
            self,
            (double, double) t_span = EMPTY_T_SPAN,
            const double[::1] y0 = None,
            tuple args = None,
            double rtol = NAN,
            double atol = NAN,
            const double[::1] rtols = None,
            const double[::1] atols = None,
            double max_step = NAN,
            double first_step = NAN,
            const double[::1] t_eval = None,
            bint auto_reset_state = True,
            bint auto_solve = False
            ):
        """
        Public method to change one or more parameters which have their own `change_*` method.
        
        See other `change_*` methods for more detailed documentation.
        
        Parameters
        ----------
        t_span
        y0
        args
        rtol
        atol
        rtols
        atols
        max_step
        first_step
        t_eval
        auto_reset_state : bint, default=True
            If True, then the `reset_state` method will be called once parameter is changed.
        auto_solve : bint, default=False
            If True, then the `solve` method will be called after all parameters have been changed and the state reset.
        """

        # This is one of the few change functions where nothing might change.
        # Track if updates need to be made
        cdef bint something_changed
        something_changed = False

        if not isnan(t_span[0]):
            something_changed = True
            self.change_t_span(t_span, auto_reset_state=False)

        if y0 is not None:
            something_changed = True
            self.change_y0(y0, auto_reset_state=False)

        if args is not None:
            something_changed = True
            self.change_args(args, auto_reset_state=False)

        if (not isnan(rtol)) or (not isnan(atol)) or (rtols is not None) or (atols is not None):
            something_changed = True
            self.change_tols(rtol=rtol, atol=atol, rtols=rtols, atols=atols, auto_reset_state=False)

        if not isnan(max_step):
            something_changed = True
            self.change_max_step(max_step, auto_reset_state=False)

        if not isnan(first_step):
            something_changed = True
            self.change_first_step(first_step, auto_reset_state=False)

        if t_eval is not None:
            something_changed = True
            self.change_t_eval(t_eval, auto_reset_state=False)

        # Now that everything has been set, reset the solver's state.
        if something_changed:
            # If first step has already been reset then no need to call it again later.
            if not isnan(first_step):
                self.recalc_first_step = False

            if auto_reset_state:
                self.reset_state()

        # User can choose to go ahead and rerun the solver with the new setup
        if auto_solve:
            # Tell solver to reset state if for some reason the user set reset to False but auto_solve to True,
            # ^ This should probably be a warning. Don't see why you'd ever want to do that.
            self._solve(reset=(not auto_reset_state))


    # Methods to be overridden by sub classes
    cdef void update_constants(self) noexcept nogil:
        # This is a template method that should be overriden by a user's subclass (if needed).

        # Example of usage:
        # If the diffeq function has an equation of the form dy = (2. * a - sin(b)) * y * sin(t)
        # then only the "y" and "sin(t)" change with each time step. The other coefficient could be precalculated to
        # save on computation steps. This method assists with that process.
        # First:
        #   Define a class attribute for the coefficient:
        # ```python
        # cdef class MySolver(CySolver):
        #     cdef double coeff_1
        # ...
        # ```
        # Second:
        #   Override this method to populate the value of `coeff_1`:
        # ```python
        # ...
        #     cdef void update_constants(self) noexcept nogil:
        #         a = self.args_ptr[0]
        #         b = self.args_ptr[1]
        #         self.coeff_1 = (2. * a - sin(b))
        # ...
        # ```
        # Third:
        #   Update the diffeq method to utilize this new coefficient variable.
        # ```python
        # ...
        #     cdef void diffeq(self) noexcept nogil:
        #         self.dy_ptr[0] = self.ceoff_1 * self.y_ptr[0] * sin(self.t_now)
        # ...
        # ```
        #
        # The `Coeff_1` variable will only be recalculated if the additional arguments are changed.

        # Base class method does nothing.
        pass

    cdef void diffeq(self) noexcept nogil:
        # This is a template function that should be overriden by the user's subclass.

        # The diffeq can use live variables which are automatically updated before each call.
        # self.t_now: The current "time" (of course, depending on your problem, it may not actually be _time_ per se).
        # self.y_ptr[:]: The current y value(s) stored as an array.
        # For example...
        # ```python
        # cdef double t_sin
        # # You will want to import the c version of sin "from libc.math cimport sin" at the top of your file.
        # t_sin = sin(self.t_now)
        # y0 = self.y_ptr[0]
        # y1 = self.y_ptr[1]
        # ```

        # Can also use other optional global attributes like...
        # self.args_ptr  (size of self.args_ptr is self.num_args). For example...
        # ```python
        # cdef double a, b
        # a = self.args_ptr[0]
        # b = self.args_ptr[1]
        # ```
        # Currently, these args must be doubles (floats).

        # This function *must* set new values to the dy_new_view variable (size of array is self.y_size). For example...
        # ```python
        # self.dy_ptr[0] = b * t_sin - y1
        # self.dy_ptr[1] = a * sin(y0)
        # ```

        # CySolver can also set additional outputs that the user may want to capture without having to make new calls
        #  to the differential equation or its sub-methods. For example...
        # ```python
        # self.extra_output_ptr[0] = t_sin
        # self.extra_output_ptr[1] = b * t_sin
        # ```
        # Currently, these additional outputs must be stored as doubles (floats).
        # Note that if extra output is used then the variables `capture_extra` and `num_extra` must be set in CySolver's
        #  `__init__` method.

        # The default template simply sets all dy to 0.
        cdef size_t i
        for i in range(self.y_size):
            self.dy_ptr[i] = 0.


    # Public accessed properties
    @property
    def t(self):
        # Need to convert the memory view back into a numpy array
        return np.ascontiguousarray(self.solution_t_view, dtype=np.float64)

    @property
    def y(self):
        # Need to convert the memory view back into a numpy array and reshape it
        return np.ascontiguousarray(self.solution_y_view, dtype=np.float64).reshape((self.len_t_touse, self.y_size)).T

    @property
    def extra(self):
        # Need to convert the memory view back into a numpy array
        return np.ascontiguousarray(self.solution_extra_view, dtype=np.float64).reshape((self.len_t_touse, self.num_extra)).T

    @property
    def growths(self):
        # How many times the output arrays had to grow during integration
        return self.num_concats - 1


    # Special methods
    def __dealloc__(self):
        # Free pointers made from user inputs
        if not (self.y0_ptr is NULL):
            PyMem_Free(self.y0_ptr)
            self.y0_ptr = NULL
        if not (self.tol_ptrs is NULL):
            PyMem_Free(self.tol_ptrs)
            self.tol_ptrs = NULL
        if not (self.args_ptr is NULL):
            PyMem_Free(self.args_ptr)
            self.args_ptr = NULL
        if not (self.t_eval_ptr is NULL):
            PyMem_Free(self.t_eval_ptr)
            self.t_eval_ptr = NULL

        # Free pointers used to track y, dydt, and any extra outputs
        if not (self.temporary_y_ptrs is NULL):
            PyMem_Free(self.temporary_y_ptrs)
            self.temporary_y_ptrs = NULL
        if not (self.extra_output_ptrs is NULL):
            PyMem_Free(self.extra_output_ptrs)
            self.extra_output_ptrs = NULL

        # Free final solution pointers
        if not (self.solution_t_ptr is NULL):
            PyMem_Free(self.solution_t_ptr)
            self.solution_t_ptr = NULL
        if not (self.solution_y_ptr is NULL):
            PyMem_Free(self.solution_y_ptr)
            self.solution_y_ptr = NULL
        if not (self.solution_extra_ptr is NULL):
            PyMem_Free(self.solution_extra_ptr)
            self.solution_extra_ptr = NULL
        
        # Free pointers used during the solve method
        if not (self._solve_time_domain_array_ptr is NULL):
            PyMem_Free(self._solve_time_domain_array_ptr)
            self._solve_time_domain_array_ptr = NULL
        if not (self._solve_y_results_array_ptr is NULL):
            PyMem_Free(self._solve_y_results_array_ptr)
            self._solve_y_results_array_ptr = NULL
        if not (self._solve_extra_array_ptr is NULL):
            PyMem_Free(self._solve_extra_array_ptr)
            self._solve_extra_array_ptr = NULL
        
        # Free pointers used during interpolation
        if not (self._interpolate_solution_t_ptr is NULL):
            PyMem_Free(self._interpolate_solution_t_ptr)
            self._interpolate_solution_t_ptr = NULL
        if not (self._interpolate_solution_y_ptr is NULL):
            PyMem_Free(self._interpolate_solution_y_ptr)
            self._interpolate_solution_y_ptr = NULL
        if not (self._interpolate_solution_extra_ptr is NULL):
            PyMem_Free(self._interpolate_solution_extra_ptr)
            self._interpolate_solution_extra_ptr = NULL

        # Free RK Temp Storage Array
        if not (self.K_ptr is NULL):
            PyMem_Free(self.K_ptr)
            self.K_ptr = NULL
