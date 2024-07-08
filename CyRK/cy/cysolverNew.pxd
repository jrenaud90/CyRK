from libcpp cimport nullptr
from libcpp cimport bool as cpp_bool
from libcpp.cmath cimport fmin, fabs

cimport cpython.ref as cpy_ref
from CyRK.utils.vector cimport vector
from CyRK.utils.memory cimport shared_ptr, make_shared


# =====================================================================================================================
# Import common functions and constants
# =====================================================================================================================
cdef extern from "common.cpp" nogil:
    const double INF
    const double EPS_100
    const unsigned int Y_LIMIT
    const unsigned int DY_LIMIT
    const double MAX_STEP

    ctypedef void (*DiffeqFuncType)(double*, double, double*, double*)

    cdef size_t find_expected_size(        
        int num_y,
        int num_extra,
        double t_delta_abs,
        double rtol_min)


# =====================================================================================================================
# Import CySolverResult (container for integration results)
# =====================================================================================================================
cdef extern from "cysolution.cpp" nogil:
    cdef cppclass CySolverResult:
            CySolverResult()
            CySolverResult(size_t num_y, size_t num_extra, size_t expected_size)
            cpp_bool success
            cpp_bool reset_called
            char* message_ptr
            int error_code
            size_t size            
            size_t num_y
            size_t num_dy
            vector[double] time_domain
            vector[double] solution
            void save_data(double new_t, double* new_solution_y, double* new_solution_dy)
            void finalize()
            void reset()
            void update_message(const char* new_message)

cdef class WrapCySolverResult:
    """ Wrapper for the C++ class `CySolverResult` defined in "cysolution.cpp" """

    cdef shared_ptr[CySolverResult] cyresult_shptr
    cdef CySolverResult* cyresult_ptr
    cdef double* time_ptr
    cdef double* y_ptr
    cdef double[::1] time_view
    cdef double[::1] y_view

    cdef size_t size
    cdef size_t num_dy

    cdef void set_cyresult_pointer(self, shared_ptr[CySolverResult] cyresult_shptr)


# =====================================================================================================================
# Import CySolver Integrator Base Class
# =====================================================================================================================
cdef extern from "cysolver.cpp" nogil:
    cdef cppclass CySolverBase:
        CySolverBase()
        CySolverBase(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const unsigned int num_y,
            const unsigned int num_extra,
            const double* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB
        )
        shared_ptr[CySolverResult] storage_ptr
        int status
        unsigned int num_y
        size_t len_t
        double t_now
        double* y0
        double* y_now_ptr
        double* dy_now_ptr
        cpp_bool check_status()
        void diffeq()
        void take_step()
        void change_storage(shared_ptr[CySolverResult] new_storage_ptr, cpp_bool auto_reset)
        void reset()
        void set_cython_extension_instance(cpy_ref.PyObject* cython_extension_class_instance)
        void py_diffeq()


cdef class WrapPyDiffeq:

    cdef object diffeq_func
    cdef tuple args
    cdef cpp_bool use_args

    cdef unsigned int num_y
    cdef unsigned int num_dy

    cdef object y_now_arr
    cdef double[::1] y_now_view

    # State attributes
    cdef double* y_now_ptr
    cdef double* t_now_ptr
    cdef double* dy_now_ptr

    cdef void set_state(self,
        double* dy_ptr,
        double* t_ptr,
        double* y_ptr
        ) noexcept nogil

# =====================================================================================================================
# Import CySolver Runge-Kutta Integrators
# =====================================================================================================================
cdef extern from "rk.cpp" nogil:
    cdef cppclass RKSolver(CySolverBase):
        RKSolver()
        RKSolver(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const unsigned int num_y,
            const unsigned int num_extra,
            const double* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
        )
        void p_step_implementation()
        void p_estimate_error()
        void reset()
        void calc_first_step_size()

    
    cdef cppclass RK23(RKSolver):
        RK23()
        RK23(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const unsigned int num_y,
            const unsigned int num_extra,
            const double* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
        )
        void reset()
    
    cdef cppclass RK45(RKSolver):
        RK45()
        RK45(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const unsigned int num_y,
            const unsigned int num_extra,
            const double* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
        )
        void reset()
    
    cdef cppclass DOP853(RKSolver):
        DOP853()
        DOP853(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const unsigned int num_y,
            const unsigned int num_extra,
            const double* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
        )
        void reset()


# =====================================================================================================================
# Import the C++ cysolve_ivp helper function
# =====================================================================================================================
cdef extern from "cysolve.cpp" nogil:
    # Pure C++ and Cython implementation
    cdef shared_ptr[CySolverResult] baseline_cysolve_ivp(
            DiffeqFuncType diffeq_ptr,
            const double* t_span_ptr,
            const double* y0_ptr,
            const size_t num_y,
            const int method,
            const size_t expected_size,
            const size_t num_extra,
            const double* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
            )

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
            shared_ptr[CySolverResult] solution_ptr,
            const double t_start,
            const double t_end,
            const double* y0_ptr,
            const unsigned int num_y,
            const unsigned int num_extra,
            const double* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size)
        PySolverStatePointers get_state_pointers()
        void solve()


# =====================================================================================================================
# Cython-based wrapper for baseline_cysolve_ivp that carries default values.
# =====================================================================================================================
cdef shared_ptr[CySolverResult] cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    double* t_span_ptr,
    double* y0_ptr,
    size_t num_y,
    int method = *,
    size_t expected_size = *,
    size_t num_extra = *,
    double* args_ptr = *,
    size_t max_num_steps = *,
    size_t max_ram_MB = *,
    double rtol = *,
    double atol = *,
    double* rtols_ptr = *,
    double* atols_ptr = *,
    double max_step_size = *,
    double first_step_size = *
    ) noexcept nogil
