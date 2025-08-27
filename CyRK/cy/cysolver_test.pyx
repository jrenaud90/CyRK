# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from libc.math cimport sin, cos
from libc.string cimport memcpy

from libcpp cimport bool as cpp_bool
from libcpp.utility cimport move
from libcpp.limits cimport numeric_limits
from libcpp.vector cimport vector

cdef double d_NAN = numeric_limits[double].quiet_NaN()

from CyRK.cy.common cimport DiffeqFuncType, MAX_STEP
from CyRK.cy.cysolver_api cimport cysolve_ivp_noreturn, cysolve_ivp, WrapCySolverResult, CySolveOutput, ODEMethod, CySolverResult
from CyRK.cy.events cimport Event

import numpy as np
cimport numpy as np
np.import_array()


cdef void baseline_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = (1. - 0.01 * y1) * y0
    dy_ptr[1] = (0.02 * y0 - 1.) * y1


cdef void accuracy_test_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = sin(t) - y1
    dy_ptr[1] = cos(t) + y0


cdef void extraoutput_test_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack y
    cdef double y0, y1, extra_0, extra_1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    extra_0 = (1. - 0.01 * y1)
    extra_1 = (0.02 * y0 - 1.)

    # Store dy/dt
    dy_ptr[0] = extra_0 * y0
    dy_ptr[1] = extra_1 * y1

    # Store extra output
    dy_ptr[2] = extra_0
    dy_ptr[3] = extra_1


cdef void lorenz_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack args
    cdef double* args_dbl_ptr = <double*>args_ptr
    cdef double a = args_dbl_ptr[0]
    cdef double b = args_dbl_ptr[1]
    cdef double c = args_dbl_ptr[2]

    # Unpack y
    cdef double y0, y1, y2
    y0 = y_ptr[0]
    y1 = y_ptr[1]
    y2 = y_ptr[2]

    dy_ptr[0] = a * (y1 - y0)
    dy_ptr[1] = y0 * (b - y2) - y1
    dy_ptr[2] = y0 * y1 - c * y2


cdef void lorenz_extraoutput_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack args
    cdef double* args_dbl_ptr = <double*>args_ptr
    cdef double a = args_dbl_ptr[0]
    cdef double b = args_dbl_ptr[1]
    cdef double c = args_dbl_ptr[2]

    # Unpack y
    cdef double y0, y1, y2
    y0 = y_ptr[0]
    y1 = y_ptr[1]
    y2 = y_ptr[2]

    cdef double e_1 = a
    cdef double e_2 = (b - y2)
    cdef double e_3 = c * y2

    dy_ptr[0] = e_1 * (y1 - y0)
    dy_ptr[1] = y0 * e_2 - y1
    dy_ptr[2] = y0 * y1 - e_3

    dy_ptr[3] = e_1
    dy_ptr[4] = e_2
    dy_ptr[5] = e_3


cdef void lotkavolterra_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack args
    cdef double* args_dbl_ptr = <double*>args_ptr
    cdef double a = args_dbl_ptr[0]
    cdef double b = args_dbl_ptr[1]
    cdef double c = args_dbl_ptr[2]
    cdef double d = args_dbl_ptr[3]

    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = a * y0 - b * y0 * y1
    dy_ptr[1] = -c * y1 + d * y0 * y1



cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack args
    cdef double* args_dbl_ptr = <double*>args_ptr
    cdef double l = args_dbl_ptr[0]
    cdef double m = args_dbl_ptr[1]
    cdef double g = args_dbl_ptr[2]

    cdef double coeff_1 = (-3. * g / (2. * l))
    cdef double coeff_2 = (3. / (m * l**2))

    # Unpack y
    cdef double y0, y1, torque
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    # External torque
    torque = 0.1 * sin(t)

    dy_ptr[0] = y1
    dy_ptr[1] = coeff_1 * sin(y0) + coeff_2 * torque


cdef struct ArbitraryArgStruct:
    double l
    # Let's make sure it has something in the middle that is not a double for size checks.
    cpp_bool cause_fail
    cpp_bool checker
    double m
    double g

cdef void arbitrary_arg_test(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Unpack args
    cdef ArbitraryArgStruct* arb_args_ptr = <ArbitraryArgStruct*>args_ptr
    cdef double l = arb_args_ptr.l
    cdef double m = arb_args_ptr.m
    cdef double g = arb_args_ptr.g
    cdef cpp_bool cause_fail = arb_args_ptr.cause_fail
    cdef cpp_bool checker    = arb_args_ptr.checker

    cdef double coeff_1 = (-3. * g / (2. * l))
    cdef double coeff_2 = (3. / (m * l**2))

    # Unpack y
    cdef double y0, y1, torque
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    # External torque
    torque = 0.1 * sin(t)

    if cause_fail:
        # Put something crazy that will cause problems
        dy_ptr[0] = (100 * t)**(1000)
        dy_ptr[1] = (-100 * t)**(1000)
    else:
        dy_ptr[0] = y1
        dy_ptr[1] = coeff_1 * sin(y0) + coeff_2 * torque



cdef void pendulum_preeval_func(char* output_ptr, double time, double* y_ptr, char* args_ptr) noexcept nogil:

    # Unpack args
    cdef double* args_dbl_ptr = <double*>args_ptr
    cdef double l = args_dbl_ptr[0]
    cdef double m = args_dbl_ptr[1]
    cdef double g = args_dbl_ptr[2]

    cdef double coeff_1 = (-3. * g / (2. * l))
    cdef double coeff_2 = (3. / (m * l**2))

    # Unpack y
    cdef double torque

    # External torque
    torque = 0.1 * sin(time)

    # Convert output pointer to double pointer so we can store data
    cdef double* output_dbl_ptr = <double*>output_ptr
    output_dbl_ptr[0] = torque
    output_dbl_ptr[1] = coeff_1
    output_dbl_ptr[2] = coeff_2


cdef void pendulum_preeval_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    
    # Make stack allocated storage for pre eval output
    cdef double[3] pre_eval_storage
    cdef double* pre_eval_storage_ptr = &pre_eval_storage[0]

    # Cast storage to void so we can call function
    cdef char* pre_eval_storage_char_ptr = <char*>pre_eval_storage_ptr

    # Call Pre-Eval Function
    pre_eval_func(pre_eval_storage_char_ptr, t, y_ptr, args_ptr)

    cdef double y0 = y_ptr[0]
    cdef double y1 = y_ptr[1]

    dy_ptr[0] = y1
    dy_ptr[1] = pre_eval_storage_ptr[1] * sin(y0) + pre_eval_storage_ptr[2] * pre_eval_storage_ptr[0]

def cy_extra_output_tester():

    cdef double t_start = 0.0
    cdef double t_end = 10.0

    cdef vector[double] y0_vec = vector[double](3)
    y0_vec[0] = 1.0
    y0_vec[1] = 0.0
    y0_vec[2] = 0.0
    
    cdef size_t num_y     = 3
    cdef size_t num_extra = 3
    cdef ODEMethod int_method = ODEMethod.RK45

    cdef size_t arg_size = sizeof(double) * 3
    cdef vector[char] args_vec = vector[char](arg_size)
    cdef double* args_ptr = <double*>args_vec.data()
    args_ptr[0] = 10.0
    args_ptr[1] = 28.0
    args_ptr[2] = 8.0 / 3.0

    cdef CySolveOutput result = cysolve_ivp(
        lorenz_extraoutput_diffeq,
        t_start,
        t_end,
        y0_vec,
        method=int_method,
        rtol=1.0e-4,
        atol=1.0e-5,
        args_vec=args_vec,
        num_extra=num_extra,
        max_num_steps=0,
        max_ram_MB=2000,
        dense_output=True
        )

    cdef double check_t = 4.335
    assert result.get().success

    # Call the result to get the baseline values for extra output
    cdef double[6] y_interp
    cdef double* y_interp_ptr = &y_interp[0]
    result.get().call(check_t, y_interp_ptr)
    cdef double dy1 = y_interp_ptr[0]
    cdef double dy2 = y_interp_ptr[1]
    cdef double dy3 = y_interp_ptr[2]
    cdef double e1  = y_interp_ptr[3]
    cdef double e2  = y_interp_ptr[4]
    cdef double e3  = y_interp_ptr[5]

    # Recall the solution to see if it still produces the expected values
    result.get().call(check_t, y_interp_ptr)

    np.testing.assert_allclose(dy1, y_interp_ptr[0])
    np.testing.assert_allclose(dy2, y_interp_ptr[1])
    np.testing.assert_allclose(dy3, y_interp_ptr[2])
    np.testing.assert_allclose(e1, y_interp_ptr[3])
    np.testing.assert_allclose(e2, y_interp_ptr[4])
    np.testing.assert_allclose(e3, y_interp_ptr[5])

    return True

def cytester(
        int diffeq_number,
        tuple t_span = None,
        const double[::1] y0 = None,
        double[::1] args = None,
        str method = 'rk45',
        size_t expected_size = 0,
        size_t max_num_steps = 0,
        size_t max_ram_MB = 2000,
        bint dense_output = False,
        double[::1] t_eval = None,
        double rtol = 1.0e-3, 
        double atol = 1.0e-6,
        double[::1] rtol_array = None,
        double[::1] atol_array = None,
        double max_step = MAX_STEP,
        double first_step = 0.0,
        WrapCySolverResult solution_reuse = None
        ):
    cdef size_t i
    cdef vector[double] t_eval_vec = vector[double]()
    if t_eval is not None:
        for i in range(t_eval.size):
            t_eval_vec.push_back(t_eval[i])

    cdef size_t num_y
    cdef vector[double] y0_vec = vector[double]()

    cdef double t_start = 0.0
    cdef double t_end   = 0.0

    # Parse method
    method = method.lower()
    cdef ODEMethod integration_method = ODEMethod.RK45
    if method == "rk23":
        integration_method = ODEMethod.RK23
    elif method == "rk45":
        integration_method = ODEMethod.RK45
    elif method == 'dop853':
        integration_method = ODEMethod.DOP853
    else:
        raise NotImplementedError(
            "ERROR: `PySolver::set_problem_parameters` - "
            f"Unknown or unsupported integration method provided: {method}.\n"
            f"Supported methods are: RK23, RK45, DOP853."
            )

    cdef size_t num_extra = 0
    cdef DiffeqFuncType diffeq = NULL
    cdef PreEvalFunc pre_eval_func = NULL
    if diffeq_number == 0:
        diffeq = baseline_diffeq
    elif diffeq_number == 1:
        diffeq = accuracy_test_diffeq
    elif diffeq_number == 2:
        diffeq = extraoutput_test_diffeq
        num_extra = 2
    elif diffeq_number == 3:
        diffeq = lorenz_diffeq
    elif diffeq_number == 4:
        diffeq = lorenz_extraoutput_diffeq
        num_extra = 3
    elif diffeq_number == 5:
        diffeq = lotkavolterra_diffeq
    elif diffeq_number == 6:
        diffeq = pendulum_diffeq
    elif diffeq_number == 7:
        diffeq = arbitrary_arg_test
    elif diffeq_number == 8:
        diffeq = pendulum_preeval_diffeq
        pre_eval_func = pendulum_preeval_func
    else:
        raise NotImplementedError

    # Set up additional argument information
    cdef vector[char] args_vec = vector[char]()
    cdef size_t size_of_args = 0
    cdef double* args_dbl_ptr = NULL
    # Abitrary arg test requires a ArbitraryArgStruct class instance to be passed in
    cdef ArbitraryArgStruct arb_arg_struct = ArbitraryArgStruct(1.0, False, True, 1.0, 9.81)
    
    # Check if generic testing was requested.
    if y0 is None:
        # Generic Testing Requested
        
        # Use a higher tolerance for testing
        rtol = 1.0e-6
        atol = 1.0e-8

        # Get inputs for requested functions
        if diffeq_number == 0:
            num_y = 2
            y0_vec.resize(num_y)
            y0_vec[0] = 20.0
            y0_vec[1] = 20.0
            t_start = 0.0
            t_end = 20.0

        elif diffeq_number == 1:
            num_y = 2
            y0_vec.resize(num_y)
            y0_vec[0] = 0.0
            y0_vec[1] = 1.0
            t_start = 0.0
            t_end = 10.0

        elif diffeq_number == 2:
            num_y = 2
            y0_vec.resize(num_y)
            y0_vec[0] = 20.0
            y0_vec[1] = 20.0
            t_start = 0.0
            t_end = 20.0

        elif diffeq_number == 3:
            num_y = 3
            y0_vec.resize(num_y)
            y0_vec[0] = 1.0
            y0_vec[1] = 0.0
            y0_vec[2] = 0.0
            t_start = 0.0
            t_end = 10.0
            args_vec.resize(3 * sizeof(double))
            args_dbl_ptr = <double*>args_vec.data()
            args_dbl_ptr[0] = 10.0
            args_dbl_ptr[1] = 28.0
            args_dbl_ptr[2] = 8.0 / 3.0
            
        elif diffeq_number == 4:
            num_y = 3
            y0_vec.resize(num_y)
            y0_vec[0] = 1.0
            y0_vec[1] = 0.0
            y0_vec[2] = 0.0
            t_start = 0.0
            t_end = 10.0
            args_vec.resize(3 * sizeof(double))
            args_dbl_ptr = <double*>args_vec.data()
            args_dbl_ptr[0] = 10.0
            args_dbl_ptr[1] = 28.0
            args_dbl_ptr[2] = 8.0 / 3.0

        elif diffeq_number == 5:
            num_y = 2
            y0_vec.resize(num_y)
            y0_vec[0] = 10.0
            y0_vec[1] = 5.0
            t_start = 0.0
            t_end = 15.0
            args_vec.resize(4 * sizeof(double))
            args_dbl_ptr = <double*>args_vec.data()
            args_dbl_ptr[0] = 1.5
            args_dbl_ptr[1] = 1.0
            args_dbl_ptr[2] = 3.0
            args_dbl_ptr[3] = 1.0

        elif diffeq_number == 6:
            num_y = 2
            y0_vec.resize(num_y)
            y0_vec[0] = 0.01
            y0_vec[1] = 0.0
            t_start = 0.0
            t_end = 10.0
            args_vec.resize(3 * sizeof(double))
            args_dbl_ptr = <double*>args_vec.data()
            args_dbl_ptr[0] = 1.0
            args_dbl_ptr[1] = 1.0
            args_dbl_ptr[2] = 9.81

        elif diffeq_number == 7:
            num_y = 2
            y0_vec.resize(num_y)
            y0_vec[0] = 0.01
            y0_vec[1] = 0.0
            t_start = 0.0
            t_end = 10.0
            # Set args pointer to our arb args struct variable's address and cast it to a void pointer
            args_vec.resize(sizeof(ArbitraryArgStruct))
            memcpy(args_vec.data(), &arb_arg_struct, sizeof(ArbitraryArgStruct))
        
        elif diffeq_number == 8:
            num_y = 2
            y0_vec.resize(num_y)
            y0_vec[0] = 0.01
            y0_vec[1] = 0.0
            t_start = 0.0
            t_end = 10.0
            args_vec.resize(3 * sizeof(double))
            args_dbl_ptr = <double*>args_vec.data()
            args_dbl_ptr[0] = 1.0
            args_dbl_ptr[1] = 1.0
            args_dbl_ptr[2] = 9.81

        else:
            raise NotImplementedError
    else:
        # Regular testing requested.
        num_y = len(y0)
        y0_vec.resize(num_y)
        for i in range(num_y):
            y0_vec[i] = y0[i]
        if t_span is not None:
            t_start = t_span[0]
            t_end   = t_span[1]
        else:
            raise AttributeError("ERROR: `cytester`: t_span not provided.")
        if args is not None:
            # This tester assumes that the args input is an array of doubles.
            args_vec.resize(args.size * sizeof(double))
            memcpy(args_vec.data(), &args[0], sizeof(double) * args.size)
        else:
            args_vec.resize(0)

    # Parse rtol
    cdef vector[double] rtols_vec = vector[double]()
    if rtol_array is not None:
        rtols_vec.resize(rtol_array.size)
        for i in range(rtol_array.size):
            rtols_vec[i] = rtol_array[i]
    
    # Parse atol
    cdef vector[double] atols_vec = vector[double]()
    if atol_array is not None:
        atols_vec.resize(atol_array.size)
        for i in range(atol_array.size):
            atols_vec[i] = atol_array[i]

    # Build python-safe solution storage
    if solution_reuse is None:
        solution_reuse = WrapCySolverResult()
        solution_reuse.build_cyresult(integration_method)
    else:
        if not solution_reuse.cyresult_uptr:
            solution_reuse.build_cyresult(integration_method)
        elif integration_method != solution_reuse.cyresult_uptr.get().integrator_method:
            raise AttributeError("Can not reuse solution that has different integration method.")

    cdef CySolverResult* solution_ptr = solution_reuse.cyresult_uptr.get()

    if not solution_ptr:
        raise AttributeError("Solution pointer not set.")

    # Setup empty events vector for these tests.
    cdef vector[Event] events_vec = vector[Event]()

    # Solve ODE
    cysolve_ivp_noreturn(
        solution_ptr,
        diffeq,
        t_start,
        t_end,
        y0_vec,
        rtol = rtol,
        atol = atol,
        args_vec = args_vec,
        num_extra = num_extra,
        max_num_steps = max_num_steps,
        max_ram_MB = max_ram_MB,
        dense_output = dense_output,
        t_eval_vec = t_eval_vec,
        pre_eval_func = pre_eval_func,
        events_vec = events_vec,
        rtols_vec = rtols_vec,
        atols_vec = atols_vec,
        max_step = max_step,
        first_step = first_step,
        expected_size = expected_size
        )
    
    solution_reuse.finalize()

    return solution_reuse
