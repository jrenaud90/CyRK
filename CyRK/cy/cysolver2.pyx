# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np

from libc.stdio cimport printf


cdef class PyCySolverResult:


    def __init__(self):

        pass

    cdef void set_cyresult_pointer(self, shared_ptr[CySolverResult] cyresult_shptr):

        # Store c++ based result and pull out key information
        self.cyresult_shptr = cyresult_shptr
        self.cyresult_ptr   = cyresult_shptr.get()
        self.size           = self.cyresult_ptr[0].size
        self.num_dy         = self.cyresult_ptr[0].num_dy

        # Convert solution to pointers and views
        self.time_ptr  = &self.cyresult_ptr[0].time_domain[0]
        self.y_ptr     = &self.cyresult_ptr[0].solution[0]
        self.time_view = <double[:self.size]>self.time_ptr
        self.y_view    = <double[:self.size * self.num_dy]>self.y_ptr

    @property
    def success(self):
        return self.cyresult_ptr.success
        
    @property
    def message(self):
        return str(self.cyresult_ptr.message_ptr, 'UTF-8')
    
    @property
    def t(self):
        return np.asarray(self.time_view, dtype=np.float64, order='C')
    
    @property
    def y(self):
        return np.asarray(self.y_view, dtype=np.float64, order='C').reshape((self.size, self.num_dy)).T
    
    @property
    def size(self):
        return self.size


## Test Diffeqs that use cython diffeqs but return python usable solutions
from CyRK.cy.diffeqs cimport (baseline_diffeq, accuracy_test_diffeq, extraoutput_test_diffeq,
    lorenz_diffeq, lorenz_extraoutput_diffeq, lotkavolterra_diffeq, pendulum_diffeq)

def test_cysolve_baseline(
        double t0 = 0.0, double tf = 50.0,
        double y0_0 = 20.0, double y0_1 = 20.0,
        int method = 1, double rtol = 1.0e-7, double atol = 1.0e-8,
        size_t max_num_steps = 100000, size_t expected_size = 0): 
    
    # Build time array
    cdef double[2] t_span
    cdef double* t_span_ptr = &t_span[0]
    t_span_ptr[0] = t0
    t_span_ptr[1] = tf

    # Build y0 array
    cdef size_t num_y = 2
    cdef double[2] y0
    cdef double* y0_ptr = &y0[0]
    y0_ptr[0] = y0_0
    y0_ptr[1] = y0_1

    cdef shared_ptr[CySolverResult] result = cysolve_ivp(
            baseline_diffeq,
            t_span_ptr,
            y0_ptr,
            num_y,
            method,
            expected_size,
            False,         # Capture extra
            0,             # Num extra
            NULL,
            max_num_steps, # max_num_steps
            2000,          # max_ram_MB
            rtol,
            atol,
            NULL,       # rtols_ptr
            NULL,       # atols_ptr
            INF,           # max_step_size
            0.0            # first_step_size
            )
        
    cdef PyCySolverResult py_result = PyCySolverResult()
    py_result.set_cyresult_pointer(result)

    return py_result


def test_cysolve_lorenz(
        double t0 = 0.0, double tf = 10.0,
        double y0_0 = 1.0, double y0_1 = 0.0, double y0_2 = 0.0,
        double arg_0 = 10.0, double arg_1 = 28.0, double arg_2 = 8.0 / 3.0,
        int method = 1, double rtol = 1.0e-3, double atol = 1.0e-6,
        size_t max_num_steps = 100000, size_t expected_size = 0): 
    
    # Build time array
    cdef double[2] t_span
    cdef double* t_span_ptr = &t_span[0]
    t_span_ptr[0] = t0
    t_span_ptr[1] = tf

    # Build y0 array
    cdef size_t num_y = 3
    cdef double[3] y0
    cdef double* y0_ptr = &y0[0]
    y0_ptr[0] = y0_0
    y0_ptr[1] = y0_1
    y0_ptr[2] = y0_2

    # Build args array
    cdef double[3] args
    cdef double* args_ptr = &args[0]
    args_ptr[0] = arg_0
    args_ptr[1] = arg_1
    args_ptr[2] = arg_2

    cdef shared_ptr[CySolverResult] result = cysolve_ivp(
            lorenz_diffeq,
            t_span_ptr,
            y0_ptr,
            num_y,
            method,
            expected_size,
            False,         # Capture extra
            0,             # Num extra
            args_ptr,
            max_num_steps, # max_num_steps
            2000,          # max_ram_MB
            rtol,
            atol,
            NULL,       # rtols_ptr
            NULL,       # atols_ptr
            INF,           # max_step_size
            0.0            # first_step_size
            )
        
    cdef PyCySolverResult py_result = PyCySolverResult()
    py_result.set_cyresult_pointer(result)

    return py_result

# def solve_ivp(
#             DiffeqFuncType diffeq_ptr,
#             tuple t_span,
#             double[::1] y0,
#             str method,
#             size_t expected_size,
#             bool capture_extra,
#             size_t num_extra,
#             tuple args,
#             size_t max_num_steps,
#             size_t max_ram_MB,
#             double rtol,
#             double atol,
#             double[::1] rtols_view,
#             double[::1] atols_view,
#             double max_step_size,
#             double first_step_size
#             ):
    

#     size_y



def test_cysolverresult():

    # No extra
    cdef CySolverResult storage = CySolverResult(2, 0, 200)

    assert not storage.success
    assert storage.size == 2

    # Try saving
    cdef double[2] y_values = [20.0, 30.0]
    cdef double time = 2.3
    cdef double[2] dy_values = [2.0, -2.0]

    storage.save_data(time, &y_values[0], &dy_values[0])
