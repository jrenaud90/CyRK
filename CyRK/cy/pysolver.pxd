from libcpp cimport bool as cpp_bool
cimport cpython.ref as cpy_ref

from CyRK.utils.memory cimport shared_ptr
from CyRK.cy.pysolver_cyhook cimport DiffeqMethod
from CyRK.cy.cysolver_api cimport CySolverResult

cimport numpy as np

# =====================================================================================================================
# Import the C++ cysolve_ivp helper function
# =====================================================================================================================
cdef extern from "cysolve.cpp" nogil:
    # Python-hook implementation
    struct PySolverStatePointers:
        double* dy_now_ptr
        double* t_now_ptr
        double* y_now_ptr

    cdef cppclass PySolver:
        PySolver()
        PySolver(
            unsigned int integration_method,
            cpy_ref.PyObject* cython_extension_class_instance,
            DiffeqMethod cython_extension_class_diffeq_method,
            shared_ptr[CySolverResult] solution_ptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const unsigned int num_y,
            const unsigned int num_extra,
            const void* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const cpp_bool dense_output,
            const double* t_eval,
            const size_t len_t_eval,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size)
        PySolverStatePointers get_state_pointers()
        void solve()


cdef class WrapPyDiffeq:

    cdef object diffeq_func
    cdef tuple args
    cdef cpp_bool use_args
    cdef cpp_bool pass_dy_as_arg

    cdef unsigned int num_y
    cdef unsigned int num_dy

    cdef np.ndarray y_now_arr
    cdef double[::1] y_now_view
    cdef np.ndarray dy_now_arr
    cdef double[::1] dy_now_view

    # State attributes
    cdef double* y_now_ptr
    cdef double* t_now_ptr
    cdef double* dy_now_ptr

    cdef void set_state(self,
        double* dy_ptr,
        double* t_ptr,
        double* y_ptr
        ) noexcept
    
    cdef void diffeq(self) noexcept
