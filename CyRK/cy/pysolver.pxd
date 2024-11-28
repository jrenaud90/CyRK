from libcpp cimport bool as cpp_bool
cimport cpython.ref as cpy_ref

from CyRK.utils.memory cimport shared_ptr
from CyRK.cy.pysolver_cyhook cimport DiffeqMethod
from CyRK.cy.cysolver_api cimport CySolverResult, NowStatePointers

cimport numpy as cnp
cnp.import_array()

# =====================================================================================================================
# Import the C++ cysolve_ivp helper function
# =====================================================================================================================
cdef extern from "cysolve.cpp" nogil:
    # Python-hook implementation

    cdef cppclass PySolver:
        PySolver()
        PySolver(
            int integration_method,
            cpy_ref.PyObject* cython_extension_class_instance,
            DiffeqMethod cython_extension_class_diffeq_method,
            shared_ptr[CySolverResult] solution_sptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const size_t num_y,
            const size_t expected_size,
            const size_t num_extra,
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
        void solve()
        int status
        int integration_method
        shared_ptr[CySolverResult] solution_sptr


cdef class WrapPyDiffeq:

    cdef object diffeq_func
    cdef tuple args
    cdef cpp_bool use_args
    cdef cpp_bool pass_dy_as_arg

    cdef size_t num_y
    cdef size_t num_dy

    cdef cnp.ndarray y_now_arr
    cdef double[::1] y_now_view
    cdef cnp.ndarray dy_now_arr
    cdef double[::1] dy_now_view

    # State attributes
    cdef double* y_now_ptr
    cdef double* t_now_ptr
    cdef double* dy_now_ptr

    cdef void set_state(self, NowStatePointers* solver_state_ptr) noexcept
    
    cdef void diffeq(self) noexcept
