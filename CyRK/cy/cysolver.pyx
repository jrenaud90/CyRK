# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import cython
cimport cython
import sys
import numpy as np
cimport numpy as np

np.import_array()

from libcpp cimport bool as bool_cpp_t
from libc.math cimport sqrt, fabs, nextafter, fmax, fmin, isnan, NAN, pow

from CyRK.array.interp cimport interp_array
from CyRK.rk.rk cimport (
    RK23_C, RK23_B, RK23_E, RK23_A, RK23_order, RK23_error_order, RK23_n_stages, RK23_LEN_C,
    RK45_C, RK45_B, RK45_E, RK45_A, RK45_order, RK45_error_order, RK45_n_stages, RK45_LEN_C,
    DOP_C_REDUCED, DOP_B, DOP_E3, DOP_E5, DOP_A_REDUCED, DOP_order, DOP_error_order, DOP_n_stages,
    DOP_n_stages_extended, DOP_LEN_C)

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

cdef (double, double) EMPTY_T_SPAN = (NAN, NAN)

# noinspection PyUnresolvedReferences
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
    solution_y_view : double[:, ::1]
        Memoryview of final solution for dependent variables.
        See Also: The public property method, `CySolver.y`
    solution_extra_view : double[:, ::1]
        Memoryview of the final solution for any extra parameters captured during integration.
        See Also: The public property method, `CySolver.extra`
    solution_t_view : double[::1]
        Memoryview of the final independent domain found during integration.
        See Also: The public property method, `CySolver.t`
    y_size : Py_ssize_t
        Number of dependent variables in system of ODEs.
    y_size_dbl : double
        Floating point version of y_size.
    y_size_sqrt : double
        Square-root of y_size.
    y0_view : double[::1]
        Memoryview of dependent variable initial conditions (y0 at t_start).
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
    direction_flag : bool_cpp_t
        If True, then integration is in the forward direction.
    num_args : Py_ssize_t
        Number of additional arguments that the `diffeq` method requires.
    arg_array_view double[::1]
        Memoryview of additional arguments used in the `diffeq` method.
    capture_extra bool_cpp_t
        Flag used if extra parameters should be captured during integration.
    num_extra Py_ssize_t
        Number of extra parameters that should be captured during integration.
    status : char; public
        Numerical flag to indicate status of integrator.
        See "Status and Error Codes.md" in the documentation for more information.
    message : str; public
        Verbal message to accompany `self.status` explaining the state (and potential errors) of the integrator.
    success : bool_cpp_t; public
        Flag indicating if the integration was successful or not.
    rtols_view : double[::1]
        Memoryview of relative tolerances for each dependent y variable.
    atols_view : double[::1]
        Memoryview of absolute tolerances for each dependent y variable.
    first_step : double
        Absolute size of the first step to be taken after t_start.
    max_step : double
        Maximum absolute step sized allowed.
    max_num_steps : Py_ssize_t
        Maximum number of steps allowed before integration auto fails.
    expected_size : Py_ssize_t
        Anticipated size of integration range, i.e., how many steps will be required.
        Used to build temporary storage arrays for the solution results.
    num_concats : Py_ssize_t
        Number of concatenations that were required during integration.
        If `expected_size` is too small then it will be expanded as needed. This variable tracks how many expansions
        were required.
        See Also: `CySolver.growths`
    recalc_first_step : bool_cpp_t
        If True, then the `first_step` size is recalculated when `reset_state` is called.
        Flag used when parameters are changed without reinitializing CySolver.
    run_interpolation : bool_cpp_t
        Flag if a final interpolation should be run once integration is completed successfully.
    interpolate_extra : bool_cpp_t
        Flag if interpolation should be run on extra parameters.
        If set to False when `run_interpolation=True`, then interpolation will be run on solution's y, t. These will
        then be used to recalculate extra parameters rather than an interpolation on the extra parameters captured
        during integration.
    len_t_eval : Py_ssize_t
        Length of user requested independent domain, `t_eval`.
    t_eval_view : double[::1]
        Memoryview of user requested independent domain, `t_eval`.
    rk_method : unsigned char
        Runge-Kutta method that will be used. Currently implemented models:
            0: ‘RK23’: Explicit Runge-Kutta method of order 3(2).
            1: ‘RK45’ (default): Explicit Runge-Kutta method of order 5(4).
            2: ‘DOP853’: Explicit Runge-Kutta method of order 8.
    rk_order : Py_ssize_t
        Runge-Kutta step power.
    error_order : Py_ssize_t
        Runge-Kutta error power.
    rk_n_stages : Py_ssize_t
        Number of Runge-Kutta stages performed for each RK step.
    rk_n_stages_plus1 : Py_ssize_t
        One more than `rk_n_stages`.
    rk_n_stages_extended : Py_ssize_t
        An extended version of `rk_n_stages` used for DOP853 method.
    error_expo : double
        Exponential used during error calculation. Utilizes `error_order`.
    len_C : Py_ssize_t
        Size of the RK C array.
    B_view : double[::1]
        Memoryview of the RK B parameter.
    E_view : double[::1]
        Memoryview of the RK E parameter.
    E3_view : double[::1]
        Memoryview of the RK E3 parameter.
    E5_view : double[::1]
        Memoryview of the RK E5 parameter.
    C_view : double[::1]
        Memoryview of the RK C parameter.
    A_view : double[:, ::1]
        Memoryview of the RK A parameter.
    K_view : double[:, ::1]
        Memoryview of the RK K parameter.
    K_T_view : double[:, ::1]
        Memoryview of the RK K parameter's transpose.
    t_new : double
        Current value of the independent variable used during integration.
    t_old : double
        Value of the independent variable at the previous step.
    step_size : double
        Current step's absolute size.
    len_t : Py_ssize_t
        Number of steps taken.
    y_new_view : double[::1]
        Current Memoryview of the dependent y variables.
    y_old_view : double[::1]
        Memoryview of the dependent y variables at the previous step.
    y_new_view : double[::1]
        Current Memoryview of dy/dt.
    y_old_view : double[::1]
        Memoryview of dy/dt at the previous step.
    extra_output_init_view : double[::1]
        Memoryview of extra outputs at the initial step (t=t0; y=y0).
        Extra outputs are parameters captured during diffeq calculation.
    extra_output_view : double[::1]
        Current Memoryview of extra outputs (at t_new).
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
            double[::1] rtols = None,
            double[::1] atols = None,
            unsigned char rk_method = 1,
            double max_step = MAX_STEP,
            double first_step = 0.,
            Py_ssize_t max_num_steps = 0,
            const double[::1] t_eval = None,
            bool_cpp_t capture_extra = False,
            Py_ssize_t num_extra = 0,
            bool_cpp_t interpolate_extra = False,
            Py_ssize_t expected_size = 0,
            bool_cpp_t auto_solve = True):
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
        auto_solve : bool_cpp_t, default=True
            If set to True, then the solver's `solve` method will be called at the end of initialization.
            Otherwise, the user will have to call `solver_instance = CySolver(...); solver_instance.solve()`
            to perform integration.
        """

        # Loop variables
        cdef Py_ssize_t i, j

        # Set integration information
        self.status  = -4  # Status code to indicate that integration has not started.
        self.message = 'Integration has not started.'
        self.success = False
        self.recalc_first_step = False

        # Store y0 values and determine y-size information
        self.y0_view     = y0
        self.y_size      = self.y0_view.size
        self.y_size_dbl  = <double>self.y_size
        self.y_size_sqrt = sqrt(self.y_size_dbl)

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
        cdef double rtol_tmp
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] rtol_array, atol_array
        rtol_array = np.empty(self.y_size, dtype=np.float64, order='C')
        atol_array = np.empty(self.y_size, dtype=np.float64, order='C')
        self.rtols_view = rtol_array
        self.atols_view = atol_array

        if rtols is not None:
            # User provided an arrayed version of rtol.
            if len(rtols) != self.y_size:
                raise AttributeError('rtol array must be the same size as y0.')
            for i in range(self.y_size):
                rtol_tmp = rtols[i]
                # Check that the tolerances are not too small.
                if rtol_tmp < EPS_100:
                    rtol_tmp = EPS_100
                self.rtols_view[i] = rtol_tmp
        else:
            # No array provided. Use the same rtol for all ys.
            # Check that the tolerances are not too small.
            if rtol < EPS_100:
                rtol = EPS_100
            for i in range(self.y_size):
                self.rtols_view[i] = rtol

        if atols is not None:
            # User provided an arrayed version of atol.
            if len(atols) != self.y_size:
                raise AttributeError('atol array must be the same size as y0.')
            for i in range(self.y_size):
                self.atols_view[i] = atols[i]
        else:
            # No array provided. Use the same atol for all ys.
            for i in range(self.y_size):
                self.atols_view[i] = atol

        # Determine maximum number of steps
        if max_num_steps == 0:
            self.max_num_steps = MAX_INT_SIZE
        elif max_num_steps < 0:
            self.status = -8
            self.message = "Attribute error."
            raise AttributeError('Negative number of max steps provided.')
        else:
            self.max_num_steps = min(max_num_steps, MAX_INT_SIZE)

        # Expected size of output arrays.
        cdef double temp_expected_size
        if expected_size == 0:
            # CySolver will attempt to guess on a best size for the arrays.
            temp_expected_size = 100. * self.t_delta_abs * fmax(1., (1.e-6 / rtol))
            temp_expected_size = fmax(temp_expected_size, 100.)
            temp_expected_size = fmin(temp_expected_size, 10_000_000.)
            self.expected_size = <Py_ssize_t>temp_expected_size
        else:
            self.expected_size = <Py_ssize_t>expected_size
        # This variable tracks how many times the storage arrays have been appended.
        # It starts at 1 since there is at least one storage array present.
        self.num_concats = 1

        # Determine optional arguments
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] arg_array
        if args is None:
            self.num_args = 0
            # Even though there are no args, initialize the array to something to avoid seg faults
            arg_array = np.empty(0, dtype=np.float64, order='C')
            self.arg_array_view = arg_array
        else:
            self.num_args = len(args)
            arg_array = np.empty(self.num_args, dtype=np.float64, order='C')
            self.arg_array_view = arg_array
            for i in range(self.num_args):
                self.arg_array_view[i] = args[i]

        # Initialize live variable arrays
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y_new, y_old, dy_new, dy_old
        y_new  = np.empty(self.y_size, dtype=np.float64, order='C')
        y_old  = np.empty(self.y_size, dtype=np.float64, order='C')
        dy_new = np.empty(self.y_size, dtype=np.float64, order='C')
        dy_old = np.empty(self.y_size, dtype=np.float64, order='C')
        self.y_new_view  = y_new
        self.y_old_view  = y_old
        self.dy_new_view = dy_new
        self.dy_old_view = dy_old

        # Determine extra outputs
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] extra_output_init, extra_output
        self.capture_extra = capture_extra
        # To avoid memory access violations we need to set the extra output arrays no matter if they are used.
        # If not used, just set them to size zero.
        if self.capture_extra:
            if num_extra == 0:
                self.status = -8
                raise AttributeError('Capture extra set to True, but number of extra set to 0.')
            self.num_extra = num_extra
        else:
            self.num_extra = 0
        extra_output_init = np.empty(self.num_extra, dtype=np.float64, order='C')
        extra_output      = np.empty(self.num_extra, dtype=np.float64, order='C')
        self.extra_output_init_view = extra_output_init
        self.extra_output_view      = extra_output

        # Determine interpolation information
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] t_eval_array
        if t_eval is None:
            self.run_interpolation = False
            self.interpolate_extra = False
            self.len_t_eval = 0
        else:
            self.run_interpolation = True
            self.interpolate_extra = interpolate_extra
            self.len_t_eval = len(t_eval)

            t_eval_array = np.empty(self.len_t_eval, dtype=np.float64, order='C')
            self.t_eval_view = t_eval_array
            for i in range(self.len_t_eval):
                self.t_eval_view[i] = t_eval[i]

        # Determine RK scheme and initialize RK memory views
        self.rk_method = rk_method
        if rk_method == 0:
            # RK23 Method
            self.rk_order    = RK23_order
            self.error_order = RK23_error_order
            self.rk_n_stages = RK23_n_stages
            self.len_C       = RK23_LEN_C
            self.A_view  = RK23_A
            self.B_view  = RK23_B
            self.C_view  = RK23_C
            self.E_view  = RK23_E

            # Unused for RK23 but initalize it anyways
            self.E3_view = RK23_E
            self.E5_view = RK23_E
        elif rk_method == 1:
            # RK45 Method
            self.rk_order    = RK45_order
            self.error_order = RK45_error_order
            self.rk_n_stages = RK45_n_stages
            self.len_C       = RK45_LEN_C
            self.A_view  = RK45_A
            self.B_view  = RK45_B
            self.C_view  = RK45_C
            self.E_view  = RK45_E

            # Unused for RK23 but initalize it anyways
            self.E3_view = RK45_E
            self.E5_view = RK45_E
        elif rk_method == 2:
            # DOP853 Method
            self.rk_order    = DOP_order
            self.error_order = DOP_error_order
            self.rk_n_stages = DOP_n_stages
            self.len_C       = DOP_LEN_C
            self.A_view  = DOP_A_REDUCED
            self.B_view  = DOP_B
            self.C_view  = DOP_C_REDUCED
            self.E3_view = DOP_E3
            self.E5_view = DOP_E5
            self.rk_n_stages_extended = DOP_n_stages_extended

            # Unused for DOP853 but initialize it anyways
            self.E_view  = DOP_E3
        else:
            self.status = -8
            self.message = "Attribute error."
            raise AttributeError(
                'Unexpected rk_method provided. Currently supported versions are:\n'
                '\t0 = RK23\n'
                '\t1 = RK34\n'
                '\t2 = DOP853')

        self.rk_n_stages_plus1 = self.rk_n_stages + 1
        self.error_expo        = 1. / (<double>self.error_order + 1.)

        # Initialize other RK-related Arrays
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] K
        # It is important K be initialized with 0s
        K = np.zeros((self.rk_n_stages_plus1, self.y_size), dtype=np.float64, order='C')
        self.K_view   = K
        self.K_T_view = self.K_view.T

        # Store user provided step information
        self.first_step = first_step
        self.max_step   = max_step

        # Parameters are initialized but may not be set to correct values.
        # Call reset state to ensure everything is ready.
        self.reset_state()

        # Run solver if requested
        if auto_solve:
            # We know for a fact that this is the first time solve will be called and we just reset the Sovler's state
            # So we can safely tell the solve method not to reset.
            self._solve(reset=False)


    cpdef void reset_state(self):
        """ Resets the class' state variables so that integration can be rerun. """
        cdef Py_ssize_t i, j

        # Set current and old time variables equal to t0
        self.t_old = self.t_start
        self.t_new = self.t_start
        self.len_t = 1

        # Reset y variables back to initial conditions
        for i in range(self.y_size):
            # Set current and old y variables equal to y0
            self.y_new_view[i] = self.y0_view[i]
            self.y_old_view[i] = self.y0_view[i]

            for j in range(self.rk_n_stages_plus1):
                # Reset RK variables
                self.K_view[j, i] = 0.

        # Update any constant parameters that the user has set
        self.update_constants()

        # Make initial call to diffeq()
        self.diffeq()

        # Store first dydt
        for i in range(self.y_size):
            self.dy_old_view[i] = self.dy_new_view[i]

        # Store extra outputs for the first time step
        if self.capture_extra:
            for i in range(self.num_extra):
                self.extra_output_init_view[i] = self.extra_output_view[i]

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
        self.num_concats = 1

        # Reset public variables to clear any old solutions
        # and to avoid memory access violations if solve() is not called.
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] solution_extra_fake, solution_y_fake
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] solution_t_fake
        solution_extra_fake = np.nan * np.ones((1, 1), dtype=np.float64, order='C')
        solution_y_fake     = np.nan * np.ones((1, 1), dtype=np.float64, order='C')
        solution_t_fake     = np.nan * np.ones(1, dtype=np.float64, order='C')
        self.solution_t_view     = solution_t_fake
        self.solution_extra_view = solution_extra_fake
        self.solution_y_view     = solution_y_fake

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
                y_old_tmp = self.y_old_view[i]

                scale = self.atols_view[i] + fabs(y_old_tmp) * self.rtols_view[i]

                d0_abs = fabs(y_old_tmp / scale)
                d1_abs = fabs(self.dy_old_view[i] / scale)
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

            self.t_new = self.t_old + h0_direction
            for i in range(self.y_size):
                self.y_new_view[i] = self.y_old_view[i] + h0_direction * self.dy_old_view[i]

            # Update dy_new_view
            self.diffeq()

            # Find the norm for d2
            d2 = 0.
            for i in range(self.y_size):

                scale = self.atols_view[i] + fabs(self.y_old_view[i]) * self.rtols_view[i]
                d2_abs = fabs( (self.dy_new_view[i] - self.dy_old_view[i]) / scale)
                d2 += (d2_abs * d2_abs)

            d2 = sqrt(d2) / (h0 * self.y_size_sqrt)

            if d1 <= 1.e-15 and d2 <= 1.e-15:
                h1 = max(1.e-6, h0 * 1.e-3)
            else:
                h1 = (0.01 / max(d1, d2))**self.error_expo

            step_size = max(10. * fabs(nextafter(self.t_old, self.direction_inf) - self.t_old),
                            min(100. * h0, h1))

        return step_size


    cdef void rk_step(self) noexcept nogil:
        """ Performs a Runge-Kutta step calculation including local error determination. """

        # Initialize step variables
        cdef Py_ssize_t s, i, j
        cdef double min_step, step, step_factor, time_tmp, t_delta_check
        cdef double scale, K_scale, dy_tmp
        cdef double error_norm3, error_norm5, error_norm, error_dot_1, error_dot_2, error_denom, error_pow
        cdef bool_cpp_t step_accepted, step_rejected, step_error

        # Run RK integration step
        # Determine step size based on previous loop
        # Find minimum step size based on the value of t (less floating point numbers between numbers when t is large)
        min_step = 10. * fabs(nextafter(self.t_old, self.direction_inf) - self.t_old)
        # Look for over/undershoots in previous step size
        if self.step_size > self.max_step:
            self.step_size = self.max_step
        elif self.step_size < min_step:
            self.step_size = min_step

        # Determine new step size
        step_accepted = False
        step_rejected = False
        step_error    = False

        # Optimization variables
        cdef double C_at_s, A_at_sj, A_at_10, B_at_j
        # Define a very specific A now since it is called consistently and does not change.
        A_at_10 = self.A_view[1, 0]

        # # Step Loop
        while not step_accepted:
            if self.step_size < min_step:
                step_error  = True
                self.status = -1
                break

            # Move time forward for this particular step size
            if self.direction_flag:
                step = self.step_size
                t_delta_check = self.t_new - self.t_end
            else:
                step = -self.step_size
                t_delta_check = self.t_end - self.t_new
            self.t_new = self.t_old + step

            # Check that we are not at the end of integration with that move
            if t_delta_check > 0.:
                self.t_new = self.t_end

                # If we are, correct the step so that it just hits the end of integration.
                step = self.t_new - self.t_old
                if self.direction_flag:
                    self.step_size = step
                else:
                    self.step_size = -step

            # # Calculate derivative using RK method

            # t_new must be updated for each loop of s in order to make the diffeq calls.
            # But we need to return to its original value later on. Store in temp variable.
            time_tmp = self.t_new

            for s in range(1, self.len_C):
                C_at_s = self.C_view[s]

                # Update t_new so it can be used in the diffeq call.
                self.t_new = self.t_old + C_at_s * step

                # Dot Product (K, a) * step
                if s == 1:
                    for i in range(self.y_size):
                        # Set the first column of K
                        dy_tmp = self.dy_old_view[i]
                        self.K_view[0, i] = dy_tmp

                        # Calculate y_new for s==1
                        self.y_new_view[i] = self.y_old_view[i] + (dy_tmp * A_at_10 * step)
                else:
                    for j in range(s):
                        A_at_sj = self.A_view[s, j]
                        for i in range(self.y_size):
                            if j == 0:
                                # Initialize
                                self.y_new_view[i] = self.y_old_view[i]

                            self.y_new_view[i] += self.K_view[j, i] * A_at_sj * step

                # Call diffeq to update K with the new dydt
                self.diffeq()

                for i in range(self.y_size):
                    self.K_view[s, i] = self.dy_new_view[i]

            # Restore t_new to its previous value.
            self.t_new = time_tmp

            # Dot Product (K, B) * step
            for j in range(self.rk_n_stages):
                B_at_j = self.B_view[j]
                # We do not use rk_n_stages_plus1 here because we are chopping off the last row of K to match
                #  the shape of B.
                for i in range(self.y_size):
                    if j == 0:
                        # Initialize
                        self.y_new_view[i] = self.y_old_view[i]

                    self.y_new_view[i] += self.K_view[j, i] * B_at_j * step

            self.diffeq()

            # Check how well this step performed by calculating its error
            if self.rk_method == 2:
                # Calculate Error for DOP853
                # Dot Product (K, E5) / scale and Dot Product (K, E3) * step / scale
                error_norm3 = 0.
                error_norm5 = 0.
                for i in range(self.y_size):
                    # Find scale of y for error calculations
                    scale = (self.atols_view[i] +
                             max(fabs(self.y_old_view[i]), fabs(self.y_new_view[i])) * self.rtols_view[i])

                    # Set last array of K equal to dydt
                    self.K_view[self.rk_n_stages, i] = self.dy_new_view[i]
                    for j in range(self.rk_n_stages_plus1):
                        if j == 0:
                            # Initialize
                            error_dot_1 = 0.
                            error_dot_2 = 0.

                        K_scale = self.K_T_view[i, j] / scale
                        error_dot_1 += K_scale * self.E3_view[j]
                        error_dot_2 += K_scale * self.E5_view[j]

                    # We need the absolute value but since we are taking the square, it is guaranteed to be positive.
                    # TODO: This will need to change if CySolver ever accepts complex numbers
                    # error_norm3_abs = fabs(error_dot_1)
                    # error_norm5_abs = fabs(error_dot_2)

                    error_norm3 += (error_dot_1 * error_dot_1)
                    error_norm5 += (error_dot_2 * error_dot_2)

                # Check if errors are zero
                if (error_norm5 == 0.) and (error_norm3 == 0.):
                    error_norm = 0.
                else:
                    error_denom = error_norm5 + 0.01 * error_norm3
                    error_norm = self.step_size * error_norm5 / sqrt(error_denom * self.y_size_dbl)

            else:
                # Calculate Error for RK23 and RK45
                # Dot Product (K, E) * step / scale
                error_norm = 0.
                for i in range(self.y_size):
                    # Find scale of y for error calculations
                    scale = (self.atols_view[i] +
                             max(fabs(self.y_old_view[i]), fabs(self.y_new_view[i])) * self.rtols_view[i])

                    # Set last array of K equal to dydt
                    self.K_view[self.rk_n_stages, i] = self.dy_new_view[i]
                    for j in range(self.rk_n_stages_plus1):
                        if j == 0:
                            # Initialize
                            error_dot_1 = 0.

                        K_scale = self.K_T_view[i, j] / scale
                        error_dot_1 += K_scale * self.E_view[j] * step

                    # We need the absolute value but since we are taking the square, it is guaranteed to be positive.
                    # TODO: This will need to change if CySolver ever accepts complex numbers
                    # error_norm_abs = fabs(error_dot_1)

                    error_norm += (error_dot_1 * error_dot_1)
                error_norm = sqrt(error_norm) / self.y_size_sqrt

            if error_norm < 1.:
                # The error is low! Let's update this step for the next time loop
                if error_norm == 0.:
                    step_factor = MAX_FACTOR
                else:
                    error_pow = pow(error_norm, -self.error_expo)
                    step_factor = min(MAX_FACTOR, SAFETY * error_pow)

                if step_rejected:
                    # There were problems with this step size on the previous step loop. Make sure factor does
                    #    not exasperate them.
                    step_factor = min(step_factor, 1.)

                self.step_size = self.step_size * step_factor
                step_accepted = True
            else:
                error_pow = pow(error_norm, -self.error_expo)
                self.step_size = self.step_size * max(MIN_FACTOR, SAFETY * error_pow)
                step_rejected = True

        if step_error:
            # Issue with step convergence
            self.status = -1
        elif not step_accepted:
            # Issue with step convergence
            self.status = -7

        # End of step loop. Update the 'old' variables
        self.t_old = self.t_new
        for i in range(self.y_size):
            self.y_old_view[i]  = self.y_new_view[i]
            self.dy_old_view[i] = self.dy_new_view[i]


    cpdef void solve(self, bool_cpp_t reset = True):
        """
        Public wrapper to the private solve method which calculates the integral of the user-provided system of ODEs.
        
        Parameters
        ----------
        reset : bool_cpp_t, default=True
            If True, `reset_state()` will be called before integration starts.
        """
        self._solve(reset=reset)


    cdef void _solve(self, bool_cpp_t reset = True):
        """
        Calculates the integral of the user-provided system of ODEs.
        
        Parameters
        ----------
        reset : bool_cpp_t, default=True
            If True, `reset_state()` will be called before integration starts.
        """

        # Reset the solver's state (avoid issues if solve() is called multiple times).
        if reset:
            self.reset_state()

        # Setup loop variables
        cdef Py_ssize_t i, j

        # Setup storage arrays
        # These arrays are built to fit a number of points equal to `self.expected_size`
        # If the integration needs more than that then a new array will be concatenated (with performance costs) to these.
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] y_results_array, extra_array
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] time_domain_array
        cdef double[:, ::1] y_results_array_view, extra_array_view
        cdef double[::1] time_domain_array_view
        y_results_array        = np.empty((self.y_size, self.expected_size), dtype=np.float64, order='C')
        time_domain_array      = np.empty(self.expected_size, dtype=np.float64, order='C')
        y_results_array_view   = y_results_array
        time_domain_array_view = time_domain_array
        if self.capture_extra:
            extra_array      = np.empty((self.num_extra, self.expected_size), dtype=np.float64, order='C')
            extra_array_view = extra_array

        # The following are unused unless the previous array size is too small to capture all of the data
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] y_results_array_new, extra_array_new
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] time_domain_array_new
        cdef double[:, ::1] y_results_array_new_view, extra_array_new_view
        cdef double[::1] time_domain_array_new_view

        # Load initial conditions into output arrays
        time_domain_array_view[0] = self.t_start
        for i in range(self.y_size):
            y_results_array_view[i, 0] = self.y0_view[i]
        if self.capture_extra:
            for i in range(self.num_extra):
                extra_array_view[i, 0] = self.extra_output_init_view[i]

        # # Main integration loop
        self.status = 0

        if self.y_size == 0:
            self.status = -6

        while self.status == 0:

            # Check that integration is not complete.
            if self.t_new == self.t_end:
                self.t_old = self.t_end
                self.status = 1
                break

            # Check that maximum number of steps has not been exceeded.
            if self.len_t > self.max_num_steps:
                if self.max_num_steps == MAX_INT_SIZE:
                    self.status = -3
                else:
                    self.status = -2
                break

            # # Perform RK Step
            self.rk_step()

            # Check if an error occurred during step calculations before storing data.
            if self.status != 0:
                break

            # Store data
            if self.len_t >= (self.num_concats * self.expected_size):
                # There is more data then we have room in our arrays.
                # Build new arrays with more space.
                # OPT: Note this is an expensive operation.
                self.num_concats += 1
                new_size = self.num_concats * self.expected_size

                time_domain_array_new = np.empty(new_size, dtype=np.float64, order='C')
                y_results_array_new   = np.empty((self.y_size, new_size), dtype=np.float64, order='C')
                time_domain_array_new_view = time_domain_array_new
                y_results_array_new_view   = y_results_array_new
                if self.capture_extra:
                    extra_array_new = np.empty((self.num_extra, new_size), dtype=np.float64, order='C')
                    extra_array_new_view = extra_array_new

                # Fill in these new arrays with the old values (new values will be added later)
                for j in range(self.y_size):
                    for i in range(self.len_t):
                        if j == 0:
                            time_domain_array_new_view[i] = time_domain_array_view[i]
                        y_results_array_new_view[j, i] = y_results_array_view[j, i]

                if self.capture_extra:
                    for j in range(self.num_extra):
                        for i in range(self.len_t):
                            extra_array_new_view[j, i] = extra_array_view[j, i]

                # No longer need the old arrays. Change where the memory-views are pointing
                y_results_array_view = y_results_array_new
                time_domain_array_view = time_domain_array_new
                # TODO: Delete the old arrays?
                if self.capture_extra:
                    extra_array_view = extra_array_new

            # There is room to add this step's results to our storage arrays.
            time_domain_array_view[self.len_t] = self.t_new
            # To match the format that scipy follows, we will take the transpose of y.
            for i in range(self.y_size):
                y_results_array_view[i, self.len_t] = self.y_new_view[i]

            if self.capture_extra:
                for i in range(self.num_extra):
                    extra_array_view[i, self.len_t] = self.extra_output_view[i]

            # Increase number of independent variable points.
            self.len_t += 1

        # Integration has stopped. Check if it was successful.
        if self.status == 1:
            self.success = True
        else:
            self.success = False

        # Create output arrays. To match the format that scipy follows, we will take the transpose of y.
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] y_results_out_array, y_results_out_array_bad
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] extra_output_out_array, extra_output_out_array_bad
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] time_domain_out_array, time_domain_out_array_bad

        if self.success:
            # Load values into output arrays.
            # The arrays built during integration likely have a bunch of unused junk at the end due to overbuilding their size.
            # This process will remove that junk and leave only the valid data.
            y_results_out_array = np.empty((self.y_size, self.len_t), dtype=np.float64, order='C')
            time_domain_out_array = np.empty(self.len_t, dtype=np.float64, order='C')
            if self.capture_extra:
                extra_output_out_array = np.empty((self.num_extra, self.len_t), dtype=np.float64, order='C')

            # Link memory views
            self.solution_y_view = y_results_out_array
            self.solution_t_view = time_domain_out_array
            if self.capture_extra:
                self.solution_extra_view = extra_output_out_array

            # Populate values
            for j in range(self.y_size):
                for i in range(self.len_t):
                    # Since this loop is only going to len_t, it will cut off any extraneous data the storage arrays
                    # hold beyond that point.
                    if j == 0:
                        self.solution_t_view[i] = time_domain_array_view[i]
                    self.solution_y_view[j, i] = y_results_array_view[j, i]
            if self.capture_extra:
                for j in range(self.num_extra):
                    for i in range(self.len_t):
                        self.solution_extra_view[j, i] = extra_array_view[j, i]
        else:
            # Integration was not successful. But we still need to build solution arrays so that accessing the solution
            # will not cause access violations.
            # Build nan arrays
            y_results_out_array_bad   = np.nan * np.ones((self.y_size, 1), dtype=np.float64, order='C')
            time_domain_out_array_bad = np.nan * np.ones(1, dtype=np.float64, order='C')
            if self.capture_extra:
                extra_output_out_array_bad = np.nan * np.ones((self.num_extra, 1), dtype=np.float64, order='C')

            # Link memory views
            self.solution_y_view = y_results_out_array_bad
            self.solution_t_view = time_domain_out_array_bad
            if self.capture_extra:
                self.solution_extra_view = extra_output_out_array_bad

        # Integration is complete. Check if interpolation was requested.
        if self.success and self.run_interpolation:
            self.interpolate()

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


    cdef void interpolate(self):
        """ Interpolate the results of a successful integration over the user provided time domain, `t_eval`. """
        # User only wants data at specific points.
        cdef char old_status
        old_status = self.status
        self.status = 2  # Interpolation is being performed.

        # Setup loop variables
        cdef Py_ssize_t i, j

        # TODO: The current version of CySolver has not implemented sicpy's dense output. Instead we use an interpolation.
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] y_results_reduced
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y_result_timeslice, y_result_temp
        y_results_reduced  = np.empty((self.y_size, self.len_t_eval), dtype=np.float64, order='C')
        y_result_timeslice = np.empty(self.len_t, dtype=np.float64, order='C')
        y_result_temp      = np.empty(self.len_t_eval, dtype=np.float64, order='C')

        cdef double[:, ::1] y_results_reduced_view
        cdef double[::1] y_result_timeslice_view, y_result_temp_view
        y_results_reduced_view  = y_results_reduced
        y_result_timeslice_view = y_result_timeslice
        y_result_temp_view      = y_result_temp

        # Create arrays for extra output which may or may not be required.
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] extra_reduced
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] extra_timeslice, extra_temp
        cdef double[:, ::1] extra_reduced_view
        cdef double[::1] extra_timeslice_view, extra_temp_view

        for j in range(self.y_size):
            # The interpolation function only works on 1D arrays, so we must loop through each of the y variables.
            # # Set timeslice equal to the time values at this y_j
            for i in range(self.len_t):
                y_result_timeslice_view[i] = self.solution_y_view[j, i]

            # Perform numerical interpolation
            interp_array(
                self.t_eval_view,
                self.solution_t_view,
                y_result_timeslice_view,
                y_result_temp_view
                )

            # Store result.
            for i in range(self.len_t_eval):
                y_results_reduced_view[j, i] = y_result_temp_view[i]

        if self.capture_extra:
            # Right now if there is any extra output then it is stored at each time step used in the RK loop.
            # We have to make a choice:
            #   - Do we interpolate the extra values that were stored?
            #   - Or do we use the interpolated t, y values to find new extra parameters at those specific points.
            # The latter method is more computationally expensive (recalls the diffeq for each y) but is more accurate.
            # This decision is set by the user with the `interpolate_extra` flag.

            # Create extra output arrays
            extra_reduced   = np.empty((self.num_extra, self.len_t_eval), dtype=np.float64, order='C')
            extra_timeslice = np.empty(self.len_t, dtype=np.float64, order='C')
            extra_temp      = np.empty(self.len_t_eval, dtype=np.float64, order='C')
            extra_reduced_view   = extra_reduced
            extra_timeslice_view = extra_timeslice
            extra_temp_view      = extra_temp

            if self.interpolate_extra:
                # Continue the interpolation for the extra values.
                for j in range(self.num_extra):
                    # np.interp only works on 1D arrays so we must loop through each of the variables:
                    # # Set timeslice equal to the time values at this y_j
                    for i in range(self.len_t):
                        extra_timeslice_view[i] = self.solution_extra_view[j, i]

                    # Perform numerical interpolation
                    interp_array(
                            self.t_eval_view,
                            self.solution_t_view,
                            extra_timeslice_view,
                            extra_temp_view
                            )

                    # Store result.
                    for i in range(self.len_t_eval):
                        extra_reduced_view[j, i] = extra_temp_view[i]
            else:
                # Use y and t to recalculate the extra outputs with self.diffeq
                for i in range(self.len_t_eval):
                    # Set state variables
                    self.t_new = self.t_eval_view[i]
                    for j in range(self.y_size):
                        self.y_new_view[j] = y_results_reduced_view[j, i]

                    # Call diffeq to recalculate extra outputs
                    self.diffeq()

                    # Capture extras
                    for j in range(self.num_extra):
                        extra_reduced_view[j, i] = self.extra_output_view[j]

        # Replace the solution variables with the new interpolated ones
        self.solution_t_view = self.t_eval_view
        self.solution_y_view = y_results_reduced_view
        if self.capture_extra:
            self.solution_extra_view = extra_reduced_view

        self.status = old_status


    cpdef void change_t_span(self, (double, double) t_span, bool_cpp_t auto_reset_state = False):
        """
        Public method to change the independent variable limits (start and stop points of integration).
        
        Parameters
        ----------
        t_span : (double, double)
            New t_span to use during integration.
        auto_reset_state : bool_cpp_t, default=False
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


    cpdef void change_y0(self, const double[::1] y0, bool_cpp_t auto_reset_state = False):
        """
        Public method to change the initial conditions.
        
        Note: the size of y0 can not be different from the original y0 used to instantiate the class instance.

        Parameters
        ----------
        y0 : double[::1]
            New dependent variable initial conditions.
            Must be the same size as the original y0.
        auto_reset_state : bool_cpp_t, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # Check y-size information
        cdef Py_ssize_t y_size_new
        y_size_new = len(y0)

        if self.y_size != y_size_new:
            # So many things need to update if ysize changes that the user should just create a new class instance.
            self.status = -8
            self.message = "Attribute error."
            raise AttributeError('New y0 must be the same size as the original y0 used to create CySolver class.'
                                 'Create new CySolver instance instead.')

        # Store y0 values for later
        self.y0_view = y0

        # A change to y0 will affect the first step's size
        self.recalc_first_step = True

        if auto_reset_state:
            self.reset_state()


    cpdef void change_args(self, tuple args, bool_cpp_t auto_reset_state = False):
        """
        Public method to change additional arguments used during integration.

        Parameters
        ----------
        args : tuple
            New tuple of additional arguments.
        auto_reset_state : bool_cpp_t, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # Determine optional arguments
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] arg_array

        self.num_args = len(args)
        arg_array = np.empty(self.num_args, dtype=np.float64, order='C')
        self.arg_array_view = arg_array
        for i in range(self.num_args):
            self.arg_array_view[i] = args[i]

        # A change to args will affect the first step's size
        self.recalc_first_step = True

        if auto_reset_state:
            self.reset_state()


    cpdef void change_tols(self, double rtol = NAN, double atol = NAN,
                           double[::1] rtols = None, double[::1] atols = None,
                           bool_cpp_t auto_reset_state = False):
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
        rtols : double[::1]
            Numpy ndarray of relative tolerances, one for each dependent y variable.
            if None (the default), then no change will be made.
        atols : double[::1]
            Numpy ndarray of absolute tolerances, one for each dependent y variable.
            if None (the default), then no change will be made.
        auto_reset_state : bool_cpp_t, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # This is one of the few change functions where nothing might change.
        # Track if updates need to be made
        cdef bool_cpp_t something_changed
        something_changed = False

        # Update tolerances
        cdef double rtol_tmp
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] rtol_array, atol_array

        if rtols is not None or not isnan(rtol):
            # Change to rtol
            something_changed = True
            rtol_array = np.empty(self.y_size, dtype=np.float64, order='C')
            self.rtols_view = rtol_array

            if rtols is not None:
                # Using arrayed rtol
                if len(rtols) != self.y_size:
                    raise AttributeError('rtols must be the same size as y0.')
                for i in range(self.y_size):
                    rtol_tmp = rtols[i]
                    if rtol_tmp < EPS_100:
                        rtol_tmp = EPS_100
                    self.rtols_view[i] = rtol_tmp
            elif not isnan(rtol):
                # Using constant rtol
                # Check tolerances
                if rtol < EPS_100:
                    rtol = EPS_100
                for i in range(self.y_size):
                    self.rtols_view[i] = rtol

        if atols is not None or not isnan(atol):
            # Change to atol
            something_changed = True
            atol_array = np.empty(self.y_size, dtype=np.float64, order='C')
            self.atols_view = atol_array

            if atols is not None:
                # Using arrayed atol
                if len(atols) != self.y_size:
                    raise AttributeError('atols must be the same size as y0.')
                for i in range(self.y_size):
                    self.atols_view[i] = atols[i]
            elif not isnan(atol):
                for i in range(self.y_size):
                    self.atols_view[i] = atol

        if something_changed:
            # A change to tolerances will affect the first step's size
            self.recalc_first_step = True

            if auto_reset_state:
                self.reset_state()


    cpdef void change_max_step(self, double max_step, bool_cpp_t auto_reset_state = False):
        """
        Public method to change maximum allowed step size.
        
        Parameters
        ----------
        max_step : double
            New maximum step size used during integration.
        auto_reset_state : bool_cpp_t, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        self.max_step = fabs(max_step)

        if auto_reset_state:
            self.reset_state()


    cpdef void change_first_step(self, double first_step, bool_cpp_t auto_reset_state = False):
        """
        Public method to change first step's size.
        
        Parameters
        ----------
        first_step : double
            New first step's size.
        auto_reset_state : bool_cpp_t, default=False
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


    cpdef void change_t_eval(self, const double[:] t_eval, bool_cpp_t auto_reset_state = False):
        """
        Public method to change user requested independent domain, `t_eval`.
        
        Parameters
        ----------
        t_eval : double[:]
            New independent domain at which solution will be interpolated.
        auto_reset_state : bool_cpp_t, default=False
            If True, then the `reset_state` method will be called once parameter is changed.
        """

        # Determine interpolation information
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] t_eval_array

        self.run_interpolation = True
        self.len_t_eval = t_eval.size

        t_eval_array = np.empty(self.len_t_eval, dtype=np.float64, order='C')
        self.t_eval_view = t_eval_array
        for i in range(self.len_t_eval):
            self.t_eval_view[i] = t_eval[i]

        if auto_reset_state:
            self.reset_state()


    cpdef void change_parameters(self,
            (double, double) t_span = EMPTY_T_SPAN,
            const double[::1] y0 = None,
            tuple args = None,
            double rtol = NAN,
            double atol = NAN,
            double[::1] rtols = None,
            double[::1] atols = None,
            double max_step = NAN,
            double first_step = NAN,
            const double[::1] t_eval = None,
            bool_cpp_t auto_reset_state = True,
            bool_cpp_t auto_solve = False):
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
        auto_reset_state : bool_cpp_t, default=True
            If True, then the `reset_state` method will be called once parameter is changed.
        auto_solve : bool_cpp_t, default=False
            If True, then the `solve` method will be called after all parameters have been changed and the state reset.
        """

        # This is one of the few change functions where nothing might change.
        # Track if updates need to be made
        cdef bool_cpp_t something_changed
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
        #         a = self.arg_array_view[0]
        #         b = self.arg_array_view[1]
        #         self.coeff_1 = (2. * a - sin(b))
        # ...
        # ```
        # Third:
        #   Update the diffeq method to utilize this new coefficient variable.
        # ```python
        # ...
        #     cdef void diffeq(self) noexcept nogil:
        #         self.dy_new_view[0] = self.ceoff_1 * self.y_new_view[0] * sin(self.t_new)
        # ...
        # ```
        #
        # The `Coeff_1` variable will only be recalculated if the additional arguments are changed.

        # Base class method does nothing.
        pass

    cdef void diffeq(self) noexcept nogil:
        # This is a template function that should be overriden by the user's subclass.

        # The diffeq can use live variables which are automatically updated before each call.
        # self.t_new: The current "time" (of course, depending on your problem, it may not actually be _time_ per se).
        # self.y_new_view[:]: The current y value(s) stored as an array.
        # For example...
        # ```python
        # cdef double t_sin
        # # You will want to import the c version of sin "from libc.math cimport sin" at the top of your file.
        # t_sin = sin(self.t_new)
        # y0 = self.y_new_view[0]
        # y1 = self.y_new_view[1]
        # ```

        # Can also use other optional global attributes like...
        # self.arg_array_view  (size of self.arg_array_view is self.num_args). For example...
        # ```python
        # cdef double a, b
        # a = self.arg_array_view[0]
        # b = self.arg_array_view[1]
        # ```
        # Currently, these args must be doubles (floats).

        # This function *must* set new values to the dy_new_view variable (size of array is self.y_size). For example...
        # ```python
        # self.dy_new_view[0] = b * t_sin - y1
        # self.dy_new_view[1] = a * sin(y0)
        # ```

        # CySolver can also set additional outputs that the user may want to capture without having to make new calls
        #  to the differential equation or its sub-methods. For example...
        # ```python
        # self.extra_output_view[0] = t_sin
        # self.extra_output_view[1] = b * t_sin
        # ```
        # Currently, these additional outputs must be stored as doubles (floats).
        # Note that if extra output is used then the variables `capture_extra` and `num_extra` must be set in CySolver's
        #  `__init__` method.

        # The default template simply sets all dy to 0.
        cdef Py_ssize_t i
        for i in range(self.y_size):
            self.dy_new_view[i] = 0.


    # Public accessed properties
    @property
    def t(self):
        # Need to convert the memory view back into a numpy array
        return np.asarray(self.solution_t_view)


    @property
    def y(self):
        # Need to convert the memory view back into a numpy array
        return np.asarray(self.solution_y_view)


    @property
    def extra(self):
        # Need to convert the memory view back into a numpy array
        return np.asarray(self.solution_extra_view)


    @property
    def growths(self):
        # How many times the output arrays had to grow during integration
        return self.num_concats - 1
