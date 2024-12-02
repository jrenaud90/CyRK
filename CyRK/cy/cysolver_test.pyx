# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libcpp cimport bool as cpp_bool
from libcpp.limits cimport numeric_limits
from libc.math cimport sin, cos
from libc.stdlib cimport malloc, free, realloc

cdef double d_NAN = numeric_limits[double].quiet_NaN()

from CyRK.cy.cysolver_api cimport cysolve_ivp, WrapCySolverResult, DiffeqFuncType,MAX_STEP, CySolveOutput

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

    cdef double[2] t_span = [0., 10.]
    cdef double* t_span_ptr = &t_span[0]

    cdef double[3] y0 = [1., 0., 0]
    cdef double* y0_ptr = &y0[0]
    
    cdef size_t num_y     = 3
    cdef size_t num_extra = 3
    cdef int int_method   = 1

    cdef size_t arg_size = sizeof(double) * 3
    cdef double* args_ptr = <double*>malloc(arg_size)
    args_ptr[0] = 10.0
    args_ptr[1] = 28.0
    args_ptr[2] = 8.0 / 3.0

    cdef CySolveOutput result = cysolve_ivp(
        lorenz_extraoutput_diffeq,
        t_span_ptr,
        y0_ptr,
        num_y,
        method=int_method,
        rtol=1.0e-4,
        atol=1.0e-5,
        args_ptr=<char*>args_ptr,
        size_of_args=arg_size,
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


    # Corrupt or otherwise change up the arg pointer
    args_ptr[0] = -99.0
    args_ptr[1] = -99.0
    args_ptr[2] = -99.0
    args_ptr = <double*>realloc(args_ptr, sizeof(double) * 50)
    cdef size_t i 
    for i in range(50):
        args_ptr[i] = -99.0
    

    # Recall the solution to see if it still produces the expected values
    result.get().call(check_t, y_interp_ptr)

    np.testing.assert_allclose(dy1, y_interp_ptr[0])
    np.testing.assert_allclose(dy2, y_interp_ptr[1])
    np.testing.assert_allclose(dy3, y_interp_ptr[2])
    np.testing.assert_allclose(e1, y_interp_ptr[3])
    np.testing.assert_allclose(e2, y_interp_ptr[4])
    np.testing.assert_allclose(e3, y_interp_ptr[5])

    free(args_ptr)

    return True

def cytester(
        int diffeq_number,
        tuple t_span = None,
        const double[::1] y0 = None,
        double[::1] args = None,
        int method = 1,
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
        double first_step = 0.0
        ):
    cdef size_t i
    cdef double* t_eval_ptr = NULL
    cdef size_t len_t_eval = 0
    if t_eval is not None:
        len_t_eval = len(t_eval)
        t_eval_ptr = &t_eval[0]

    cdef size_t num_y
    cdef double[10] y0_arr
    cdef double* y0_ptr = &y0_arr[0]

    cdef double[2] t_span_arr
    cdef double* t_span_ptr = &t_span_arr[0]

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
    cdef char* args_ptr = NULL
    cdef size_t size_of_args = 0

    # Arg double array for the diffeq's that use it
    cdef bint cast_arg_dbl   = False
    cdef double[10] args_arr
    cdef double* args_ptr_dbl = &args_arr[0]
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
            y0_ptr[0] = 20.0
            y0_ptr[1] = 20.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 20.0

        elif diffeq_number == 1:
            num_y = 2
            y0_ptr[0] = 0.0
            y0_ptr[1] = 1.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 10.0

        elif diffeq_number == 2:
            num_y = 2
            y0_ptr[0] = 20.0
            y0_ptr[1] = 20.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 20.0

        elif diffeq_number == 3:
            num_y = 3
            y0_ptr[0] = 1.0
            y0_ptr[1] = 0.0
            y0_ptr[2] = 0.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 10.0
            args_ptr_dbl[0] = 10.0
            args_ptr_dbl[1] = 28.0
            args_ptr_dbl[2] = 8.0 / 3.0
            cast_arg_dbl = True
            
        elif diffeq_number == 4:
            num_y = 3
            y0_ptr[0] = 1.0
            y0_ptr[1] = 0.0
            y0_ptr[2] = 0.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 10.0
            args_ptr_dbl[0] = 10.0
            args_ptr_dbl[1] = 28.0
            args_ptr_dbl[2] = 8.0 / 3.0
            cast_arg_dbl = True

        elif diffeq_number == 5:
            num_y = 2
            y0_ptr[0] = 10.0
            y0_ptr[1] = 5.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 15.0
            args_ptr_dbl[0] = 1.5
            args_ptr_dbl[1] = 1.0
            args_ptr_dbl[2] = 3.0
            args_ptr_dbl[3] = 1.0
            cast_arg_dbl = True

        elif diffeq_number == 6:
            num_y = 2
            y0_ptr[0] = 0.01
            y0_ptr[1] = 0.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 10.0
            args_ptr_dbl[0] = 1.0
            args_ptr_dbl[1] = 1.0
            args_ptr_dbl[2] = 9.81
            cast_arg_dbl = True

        elif diffeq_number == 7:
            num_y = 2
            y0_ptr[0] = 0.01
            y0_ptr[1] = 0.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 10.0
            # Set args pointer to our arb args struct variable's address and cast it to a void pointer
            args_ptr = <char*>&arb_arg_struct
            size_of_args = sizeof(arb_arg_struct)
            cast_arg_dbl = False
        
        elif diffeq_number == 8:
            num_y = 2
            y0_ptr[0] = 0.01
            y0_ptr[1] = 0.0
            t_span_ptr[0] = 0.0
            t_span_ptr[1] = 10.0
            args_ptr_dbl[0] = 1.0
            args_ptr_dbl[1] = 1.0
            args_ptr_dbl[2] = 9.81
            cast_arg_dbl = True

        else:
            raise NotImplementedError
    else:
        # Regular testing requested.
        num_y = len(y0)
        for i in range(num_y):
            y0_ptr[i] = y0[i]
        t_span_ptr[0] = t_span[0]
        t_span_ptr[1] = t_span[1]
        if args is not None:
            args_ptr     = <char*>&args[0]
            size_of_args = sizeof(double) # * args.size
        else:
            args_ptr     = NULL
            size_of_args = 0

    if cast_arg_dbl:
        args_ptr     = <char*>args_ptr_dbl
        size_of_args = sizeof(args_arr)

    # Parse rtol
    cdef double* rtols_ptr = NULL
    if rtol_array is not None:
        rtols_ptr = &rtol_array[0]
    
    # Parse atol
    cdef double* atols_ptr = NULL
    if atol_array is not None:
        atols_ptr = &atol_array[0]

    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_span_ptr,
        y0_ptr,
        num_y,
        method = method,
        rtol = rtol,
        atol = atol,
        args_ptr = args_ptr,
        size_of_args = size_of_args,
        num_extra = num_extra,
        max_num_steps = max_num_steps,
        max_ram_MB = max_ram_MB,
        dense_output = dense_output,
        t_eval = t_eval_ptr,
        len_t_eval = len_t_eval,
        pre_eval_func = pre_eval_func,
        rtols_ptr = rtols_ptr,
        atols_ptr = atols_ptr,
        max_step = max_step,
        first_step = first_step,
        expected_size = expected_size
        )
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(result)

    return pysafe_result
