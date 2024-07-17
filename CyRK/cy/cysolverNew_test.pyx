# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.math cimport sin, cos, fabs, fmin, fmax

from CyRK.cy.cysolverNew cimport (
    cysolve_ivp, find_expected_size, WrapCySolverResult, DiffeqFuncType,MAX_STEP, EPS_100, INF,
    CySolverResult, CySolveOutput
    )
from CyRK.utils.memory cimport shared_ptr

import numpy as np


cdef void baseline_diffeq(double* dy_ptr, double t, double* y_ptr, const double* args_ptr) noexcept nogil:
    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = (1. - 0.01 * y1) * y0
    dy_ptr[1] = (0.02 * y0 - 1.) * y1


cdef void accuracy_test_diffeq(double* dy_ptr, double t, double* y_ptr, const double* args_ptr) noexcept nogil:
    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = sin(t) - y1
    dy_ptr[1] = cos(t) + y0


cdef void extraoutput_test_diffeq(double* dy_ptr, double t, double* y_ptr, const double* args_ptr) noexcept nogil:
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


cdef void lorenz_diffeq(double* dy_ptr, double t, double* y_ptr, const double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double a = args_ptr[0]
    cdef double b = args_ptr[1]
    cdef double c = args_ptr[2]

    # Unpack y
    cdef double y0, y1, y2
    y0 = y_ptr[0]
    y1 = y_ptr[1]
    y2 = y_ptr[2]

    dy_ptr[0] = a * (y1 - y0)
    dy_ptr[1] = y0 * (b - y2) - y1
    dy_ptr[2] = y0 * y1 - c * y2


cdef void lorenz_extraoutput_diffeq(double* dy_ptr, double t, double* y_ptr, const double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double a = args_ptr[0]
    cdef double b = args_ptr[1]
    cdef double c = args_ptr[2]

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


cdef void lotkavolterra_diffeq(double* dy_ptr, double t, double* y_ptr, const double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double a = args_ptr[0]
    cdef double b = args_ptr[1]
    cdef double c = args_ptr[2]
    cdef double d = args_ptr[3]

    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = a * y0 - b * y0 * y1
    dy_ptr[1] = -c * y1 + d * y0 * y1



cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, const double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double l = args_ptr[0]
    cdef double m = args_ptr[1]
    cdef double g = args_ptr[2]

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


def cytester(
        int diffeq_number,
        tuple t_span,
        const double[::1] y0,
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
    
    cdef double[2] t_span_arr
    cdef double* t_span_ptr = &t_span_arr[0]
    t_span_ptr[0] = t_span[0]
    t_span_ptr[1] = t_span[1]

    cdef double* t_eval_ptr = NULL
    cdef size_t len_t_eval = 0
    if t_eval is not None:
        len_t_eval = len(t_eval)
        t_eval_ptr = &t_eval[0]

    cdef unsigned int num_y   = len(y0)
    cdef const double* y0_ptr = &y0[0]

    cdef int num_extra = 0
    cdef DiffeqFuncType diffeq
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
    else:
        raise NotImplementedError

    cdef double* args_ptr = NULL
    if args is not None:
        args_ptr = &args[0]

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
        num_extra = num_extra,
        max_num_steps = max_num_steps,
        max_ram_MB = max_ram_MB,
        dense_output = dense_output,
        t_eval = t_eval_ptr,
        len_t_eval = len_t_eval,
        rtols_ptr = rtols_ptr,
        atols_ptr = atols_ptr,
        max_step = max_step,
        first_step = first_step,
        expected_size = expected_size
        )
    
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(result)

    return pysafe_result
