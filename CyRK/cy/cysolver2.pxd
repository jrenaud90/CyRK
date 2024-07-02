from libcpp cimport nullptr
from libcpp cimport bool as cpp_bool

from CyRK.utils.vector cimport vector
from CyRK.utils.memory cimport shared_ptr


ctypedef void (*DiffeqFuncType)(double*, double, double*, double*) noexcept nogil


cdef extern from "common.cpp" nogil:
    const double INF

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
            void reset()
            void update_message(const char* new_message)

cdef extern from "cysolver.cpp" nogil:
    cdef cppclass CySolverBase:
        CySolverBase()
        CySolverBase(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            double* y0_ptr,
            size_t num_y,
            cpp_bool capture_extra,
            size_t num_extra,
            double* args_ptr,
            size_t max_num_steps,
            size_t max_ram_MB
        )
        shared_ptr[CySolverResult] storage_ptr
        int status
        size_t num_y
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


cdef extern from "rk.cpp" nogil:
    cdef cppclass RKSolver(CySolverBase):
        RKSolver()
        RKSolver(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            double* y0_ptr,
            size_t num_y,
            cpp_bool capture_extra,
            size_t num_extra,
            double* args_ptr,
            size_t max_num_steps,
            size_t max_ram_MB,
            double rtol,
            double atol,
            double* rtols_ptr,
            double* atols_ptr,
            double max_step_size,
            double first_step_size
        )
    
    cdef cppclass RK23(RKSolver):
        RK23()
        RK23(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            double* y0_ptr,
            size_t num_y,
            cpp_bool capture_extra,
            size_t num_extra,
            double* args_ptr,
            size_t max_num_steps,
            size_t max_ram_MB,
            double rtol,
            double atol,
            double* rtols_ptr,
            double* atols_ptr,
            double max_step_size,
            double first_step_size
        )
    
    cdef cppclass RK45(RKSolver):
        RK45()
        RK45(
            DiffeqFuncType diffeq_ptr,
            shared_ptr[CySolverResult] storage_ptr,
            const double t_start,
            const double t_end,
            double* y0_ptr,
            size_t num_y,
            cpp_bool capture_extra,
            size_t num_extra,
            double* args_ptr,
            size_t max_num_steps,
            size_t max_ram_MB,
            double rtol,
            double atol,
            double* rtols_ptr,
            double* atols_ptr,
            double max_step_size,
            double first_step_size
        )


cdef extern from "cysolve.cpp" nogil:
    cdef shared_ptr[CySolverResult] cysolve_ivp(
            DiffeqFuncType diffeq_ptr,
            double* t_span_ptr,
            double* y0_ptr,
            size_t num_y,
            int method,
            size_t expected_size,
            cpp_bool capture_extra,
            size_t num_extra,
            double* args_ptr,
            size_t max_num_steps,
            size_t max_ram_MB,
            double rtol,
            double atol,
            double* rtols_ptr,
            double* atols_ptr,
            double max_step_size,
            double first_step_size
            )


cdef class PyCySolverResult:

    cdef shared_ptr[CySolverResult] cyresult_shptr
    cdef CySolverResult* cyresult_ptr
    cdef double* time_ptr
    cdef double* y_ptr
    cdef double[::1] time_view
    cdef double[::1] y_view

    cdef size_t size
    cdef size_t num_dy

    cdef void set_cyresult_pointer(self, shared_ptr[CySolverResult] cyresult_shptr)
