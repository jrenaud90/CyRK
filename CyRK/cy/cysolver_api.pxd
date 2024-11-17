from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cimport cpython.ref as cpy_ref
from CyRK.cy.pysolver_cyhook cimport DiffeqMethod

cimport numpy as cnp
cnp.import_array()


# =====================================================================================================================
# Import common functions and constants
# =====================================================================================================================
cdef extern from "common.cpp" nogil:
    const double INF
    const double EPS_100
    const unsigned int Y_LIMIT
    const unsigned int DY_LIMIT
    const double MAX_STEP

    ctypedef void (*PreEvalFunc)(void*, double, double*, const void*)
    ctypedef void (*DiffeqFuncType)(double*, double, double*, const void*, PreEvalFunc)

    cdef size_t find_expected_size(        
        int num_y,
        int num_extra,
        double t_delta_abs,
        double rtol_min)


cdef extern from "cy_array.cpp" nogil:
    size_t binary_search_with_guess(double key, const double* array, size_t length, size_t guess)


cdef extern from "dense.cpp" nogil:
    cdef cppclass CySolverDense:
        CySolverDense()
        CySolverDense(
            int integrator_int,
            shared_ptr[CySolverBase] solver_sptr,
            cpp_bool set_state)

        int integrator_int
        unsigned int num_y
        unsigned int num_extra
        shared_ptr[CySolverBase] solver_sptr
        double t_old
        double t_now
        double step
        unsigned int Q_order

        void set_state()
        void call(double t_interp, double* y_interped)


# =====================================================================================================================
# Import CySolverResult (container for integration results)
# =====================================================================================================================
cdef extern from "cysolution.cpp" nogil:
    cdef cppclass CySolverResult:
            CySolverResult()
            CySolverResult(
                const size_t num_y,
                const size_t num_extra,
                const size_t expected_size,
                const size_t last_t,
                const cpp_bool direction_flag,
                const cpp_bool capture_dense_output,
                const cpp_bool t_eval_provided)
            
            cpp_bool capture_extra
            cpp_bool retain_solver
            cpp_bool capture_dense_output
            cpp_bool t_eval_provided
            cpp_bool success
            cpp_bool reset_called
            cpp_bool direction_flag
            int error_code
            unsigned int integrator_method
            unsigned int num_y
            unsigned int num_extra
            unsigned int num_dy
            char* message_ptr
            size_t size
            size_t num_interpolates
            vector[double] time_domain_vec
            vector[double] time_domain_vec_sorted
            vector[double] solution
            double* time_domain_vec_sorted_ptr
            vector[CySolverDense] dense_vec
            shared_ptr[CySolverBase] solver_sptr
            vector[double] interp_time_vec

            void save_data(double new_t, double* new_solution_y, double* new_solution_dy)
            CySolverDense* build_dense(cpp_bool save)
            void solve()
            void finalize()
            void reset()
            void build_solver(
                DiffeqFuncType diffeq_ptr,
                const double t_start,
                const double t_end,
                const double* y0_ptr,
                const unsigned int method,
                const size_t expected_size,
                const void* args_ptr,
                const size_t max_num_steps,
                const size_t max_ram_MB,
                const cpp_bool dense_output,
                const double* t_eval,
                const size_t len_t_eval,
                PreEvalFunc pre_eval_func,
                const double rtol,
                const double atol,
                const double* rtols_ptr,
                const double* atols_ptr,
                const double max_step_size,
                const double first_step_size
            )
            void update_message(const char* new_message)
            void call(const double t, double* y_interp)
            void call_vectorize(const double* t_array_ptr, size_t len_t, double* y_interp)

ctypedef shared_ptr[CySolverResult] CySolveOutput

cdef class WrapCySolverResult:
    """ Wrapper for the C++ class `CySolverResult` defined in "cysolution.cpp" """

    cdef shared_ptr[CySolverResult] cyresult_shptr
    cdef CySolverResult* cyresult_ptr
    cdef double* time_ptr
    cdef double* y_ptr
    cdef double[::1] time_view
    cdef double[::1] y_view

    cdef void set_cyresult_pointer(self, shared_ptr[CySolverResult] cyresult_shptr)


# =====================================================================================================================
# Import CySolver Integrator Base Class
# =====================================================================================================================
cdef extern from "cysolver.cpp" nogil:
    struct NowStatePointers:
        double* t_now_ptr
        double* y_now_ptr
        double* dy_now_ptr

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
            const void* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const cpp_bool use_dense_output,
            const double* t_eval,
            const size_t len_t_eval,
            PreEvalFunc pre_eval_func
        )

        cpp_bool use_pysolver
        DiffeqMethod py_diffeq_method
        cpy_ref.PyObject cython_extension_class_instance
        int status
        unsigned int integration_method
        unsigned int num_dy
        unsigned int num_y
        shared_ptr[CySolverResult] storage_ptr
        size_t len_t
        double t_now
        double* y_now_ptr
        double* dy_now_ptr

        cpp_bool check_status()
        NowStatePointers get_now_state()
        void reset()
        void offload_to_temp()
        void load_back_from_temp()
        void calc_first_step_size()
        void take_step()
        void solve()
        void cy_diffeq()
        void diffeq()
        void set_cython_extension_instance(cpy_ref.PyObject* cython_extension_class_instance)
        void py_diffeq()


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
            const void* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const cpp_bool use_dense_output,
            const double* t_eval,
            const size_t len_t_eval,
            PreEvalFunc pre_eval_func,
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


    cdef unsigned int RK23_METHOD_INT
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
            const void* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const cpp_bool use_dense_output,
            const double* t_eval,
            const size_t len_t_eval,
            PreEvalFunc pre_eval_func,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
        )
        void reset()
    
    cdef unsigned int RK45_METHOD_INT
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
            const void* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const cpp_bool use_dense_output,
            const double* t_eval,
            const size_t len_t_eval,
            PreEvalFunc pre_eval_func,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
        )
        void reset()
    
    cdef unsigned int DOP853_METHOD_INT
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
            const void* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const cpp_bool use_dense_output,
            const double* t_eval,
            const size_t len_t_eval,
            PreEvalFunc pre_eval_func,
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
            const unsigned int num_y,
            const unsigned int method,
            const size_t expected_size,
            const unsigned int num_extra,
            const void* args_ptr,
            const size_t max_num_steps,
            const size_t max_ram_MB,
            const cpp_bool dense_output,
            const double* t_eval,
            const size_t len_t_eval,
            PreEvalFunc pre_eval_func,
            const double rtol,
            const double atol,
            const double* rtols_ptr,
            const double* atols_ptr,
            const double max_step_size,
            const double first_step_size
            )


# =====================================================================================================================
# Cython-based wrapper for baseline_cysolve_ivp that carries default values.
# =====================================================================================================================
cdef CySolveOutput cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const unsigned int num_y,
    unsigned int method = *,
    double rtol = *,
    double atol = *,
    void* args_ptr = *,
    unsigned int num_extra = *,
    size_t max_num_steps = *,
    size_t max_ram_MB = *,
    bint dense_output = *,
    double* t_eval = *,
    size_t len_t_eval = *,
    PreEvalFunc pre_eval_func = *,
    double* rtols_ptr = *,
    double* atols_ptr = *,
    double max_step = *,
    double first_step = *,
    size_t expected_size = *
    ) noexcept nogil

cdef CySolveOutput cysolve_ivp_gil(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const unsigned int num_y,
    unsigned int method = *,
    double rtol = *,
    double atol = *,
    void* args_ptr = *,
    unsigned int num_extra = *,
    size_t max_num_steps = *,
    size_t max_ram_MB = *,
    bint dense_output = *,
    double* t_eval = *,
    size_t len_t_eval = *,
    PreEvalFunc pre_eval_func = *,
    double* rtols_ptr = *,
    double* atols_ptr = *,
    double max_step = *,
    double first_step = *,
    size_t expected_size = *
    ) noexcept
