from libcpp cimport bool as cpp_bool
cimport cpython.ref as cpy_ref

from CyRK.utils.memory cimport unique_ptr, make_unique
from CyRK.cy.pysolver_cyhook cimport DiffeqMethod, PyEventMethod
from CyRK.cy.cysolver_api cimport ODEMethod, CySolverResult, NowStatePointers, WrapCySolverResult

cimport numpy as cnp
cnp.import_array()

# =====================================================================================================================
# Import the C++ cysolve_ivp helper function
# =====================================================================================================================
cdef class PySolver(WrapCySolverResult):

    cdef object diffeq_func
    cdef tuple args
    cdef cpp_bool use_args
    cdef cpp_bool pass_dy_as_arg

    cdef size_t num_y
    cdef size_t num_dy

    cdef cnp.ndarray y_tmp_arr
    cdef double[::1] y_tmp_view
    cdef cnp.ndarray y_now_arr
    cdef double[::1] y_now_view
    cdef cnp.ndarray dy_now_arr
    cdef double[::1] dy_now_view

    # State attributes
    cdef double* y_now_ptr
    cdef double* t_now_ptr
    cdef double* dy_now_ptr

    # Event data
    cdef list events_list

    cdef void set_state(self, NowStatePointers* solver_state_ptr) noexcept
    cdef void diffeq(self) noexcept
    cdef double check_pyevent(
        self,
        size_t event_index,
        double t, 
        double* y_ptr) noexcept
