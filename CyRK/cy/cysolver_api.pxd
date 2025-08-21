from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string as cpp_string

cimport cpython.ref as cpy_ref
from CyRK.cy.pysolver_cyhook cimport DiffeqMethod

cimport numpy as cnp
cnp.import_array()


# =====================================================================================================================
# Import common functions and constants
# =====================================================================================================================
cdef extern from "common.cpp" nogil:
    cpdef enum class CyrkErrorCodes(int):
        CONVERGED,
        INITIALIZING,
        SUCCESSFUL_INTEGRATION,
        NO_ERROR,
        GENERAL_ERROR,
        PROPERTY_NOT_SET,
        UNSUPPORTED_UNKNOWN_MODEL,
        UNINITIALIZED_CLASS,
        CYSOLVER_INITIALIZATION_ERROR,
        INCOMPATIBLE_INPUT,
        ATTRIBUTE_ERROR,
        BOUNDS_ERROR,
        ARGUMENT_NOT_SET,
        ARGUMENT_ERROR,
        SETUP_NOT_CALLED,
        DENSE_OUTPUT_NOT_SAVED,
        BAD_CONFIG_DATA,
        OPTIMIZE_SIGN_ERROR,
        OPTIMIZE_CONVERGENCE_ERROR,
        MEMORY_ALLOCATION_ERROR,
        VECTOR_SIZE_EXCEEDS_LIMITS,
        NUMBER_OF_EQUATIONS_IS_ZERO,
        MAX_ITERATIONS_HIT,
        MAX_STEPS_USER_EXCEEDED,
        MAX_STEPS_SYSARCH_EXCEEDED,
        STEP_SIZE_ERROR_SPACING,
        STEP_SIZE_ERROR_ACCEPTANCE,
        DENSE_BUILD_FAILED,
        INTEGRATION_NOT_SUCCESSFUL,
        ERROR_IMPORTING_PYTHON_MODULE,
        BAD_INITIAL_STEP_SIZE,
        OTHER_ERROR,
        UNSET_ERROR_CODE
    const cpp_map[CyrkErrorCodes, cpp_string] CyrkErrorMessages
    
    const double INF
    const double EPS_100
    const size_t BUFFER_SIZE
    const double MAX_STEP

    ctypedef void (*PreEvalFunc)(char*, double, double*, char*)
    ctypedef void (*DiffeqFuncType)(double*, double, double*, char*, PreEvalFunc)

    cdef void round_to_2(size_t& initial_value) noexcept

    cdef size_t find_expected_size(
        size_t num_y,
        size_t num_extra,
        double t_delta_abs,
        double rtol_min)


cdef extern from "cy_array.cpp" nogil:
    size_t binary_search_with_guess(double key, const double* array, size_t length, size_t guess)


cdef extern from "dense.cpp" nogil:
    cdef cppclass CySolverDense:
        CySolverDense()
        CySolverDense(
            CySolverBase* solver_ptr,
            cpp_bool set_state)

        void set_state()
        void call(double t_interp, double* y_interped)


# =====================================================================================================================
# Import CySolverResult (container for integration results)
# =====================================================================================================================
cdef extern from "cysolution.cpp" nogil:
    cdef cppclass CySolverResult:
            CySolverResult()
            CySolverResult(ODEMethod integration_method_)
            
            cpp_string message
            unique_ptr[ProblemConfig] config_uptr
            ODEMethod integrator_method
            CyrkErrorCodes status
            size_t size
            size_t num_interpolates
            size_t steps_taken
            cpp_bool setup_called
            cpp_bool success
            cpp_bool retain_solver
            cpp_bool capture_dense_output
            cpp_bool capture_extra
            cpp_bool t_eval_provided
            cpp_bool direction_flag
            size_t num_y
            size_t num_dy
            vector[double] time_domain_vec
            vector[double] solution
            vector[double] time_domain_vec_sorted
            vector[double]* time_domain_vec_sorted_ptr
            vector[CySolverDense] dense_vec
            unique_ptr[CySolverBase] solver_uptr
            vector[double] interp_time_vec

            void update_status(CyrkErrorCodes status_code)
            CyrkErrorCodes setup()
            CyrkErrorCodes setup(ProblemConfig* config_ptr)
            void save_data(double new_t, double* new_solution_y, double* new_solution_dy) noexcept
            int build_dense(cpp_bool save) noexcept
            CyrkErrorCodes solve()
            CyrkErrorCodes call(const double t, double* y_interp)
            CyrkErrorCodes call_vectorize(const double* t_array_ptr, size_t len_t, double* y_interp)

ctypedef unique_ptr[CySolverResult] CySolveOutput

cdef class WrapCySolverResult:
    """ Wrapper for the C++ class `CySolverResult` defined in "cysolution.cpp" """

    cdef CySolveOutput cyresult_uptr
    cdef double* time_ptr
    cdef double* y_ptr
    cdef double[::1] time_view
    cdef double[::1] y_view

    cdef void build_cyresult(self, ODEMethod integrator_method)
    cdef void set_cyresult_pointer(self, CySolveOutput cyresult_uptr_)
    cdef set_problem_config(self, ProblemConfig* new_problem_config_ptr)
    cpdef solve(self)
    cpdef finalize(self)

# =====================================================================================================================
# Import CySolver Integrator Base Class
# =====================================================================================================================
cdef extern from "cysolver.cpp" nogil:
    cpdef enum class ODEMethod(int):
        NO_METHOD_SET,
        BASE_METHOD,
        RK_BASE_METHOD,
        RK23,
        RK45,
        DOP853
    const cpp_map[ODEMethod, cpp_string] CyrkODEMethods

    cdef cppclass ProblemConfig:
        ProblemConfig()
        ProblemConfig(
            DiffeqFuncType diffeq_ptr_,
            double t_start_,
            double t_end_,
            vector[double]& y0_vec_)
        ProblemConfig(
            DiffeqFuncType diffeq_ptr_,
            double t_start_,
            double t_end_,
            vector[double]& y0_vec_,
            vector[char]& args_vec_,
            vector[double]& t_eval_vec_,
            size_t num_extra_,
            size_t expected_size_,
            size_t max_num_steps_,
            size_t max_ram_MB_,
            PreEvalFunc pre_eval_func_,
            cpp_bool capture_dense_output_,
            cpp_bool force_retain_solver_)
        
        DiffeqFuncType diffeq_ptr
        double t_start
        double t_end
        vector[double] y0_vec
        vector[char] args_vec
        vector[double] t_eval_vec
        size_t num_extra
        size_t expected_size
        size_t max_num_steps
        size_t max_ram_MB
        PreEvalFunc pre_eval_func
        cpp_bool capture_dense_output
        cpp_bool force_retain_solver
        cpp_bool capture_extra
        cpp_bool t_eval_provided
        size_t num_y
        double num_y_dbl
        double num_y_sqrt
        size_t num_dy
        double num_dy_dbl
        cpy_ref.PyObject* cython_extension_class_instance
        DiffeqMethod py_diffeq_method
        cpp_bool initialized

        void update_properties(
            DiffeqFuncType diffeq_ptr_,
            double t_start_,
            double t_end_,
            vector[double]& y0_vec_
        )
        void update_properties(
            DiffeqFuncType diffeq_ptr_,
            double t_start_,
            double t_end_,
            vector[double]& y0_vec_,
            vector[char]& args_vec_,
            vector[double]& t_eval_vec_,
            size_t num_extra_,
            size_t expected_size_,
            size_t max_num_steps_,
            size_t max_ram_MB_,
            PreEvalFunc pre_eval_func_,
            cpp_bool capture_dense_output_,
            cpp_bool force_retain_solver_
        )
        void initialize()
        void update_properties_from_config(ProblemConfig* new_config_ptr)

    struct NowStatePointers:
        double* t_now_ptr
        double* y_now_ptr
        double* dy_now_ptr

    cdef cppclass CySolverBase:
        CySolverBase()
        CySolverBase(CySolverResult* storage_ptr_)

        cpp_bool use_dense_output
        cpp_bool user_provided_max_num_steps
        cpp_bool use_pysolver
        ODEMethod integration_method
        size_t num_y
        size_t num_extra
        size_t num_dy
        double t_old
        double t_now
        double* y_old_ptr
        double* y_now_ptr
        double* dy_now_ptr

        void set_Q_order(size_t* Q_order_ptr)
        void set_Q_array(double* Q_ptr)
        void clear_python_refs()
        void offload_to_temp() noexcept
        void load_back_from_temp() noexcept
        CyrkErrorCodes resize_num_y(size_t num_y_, size_t num_dy_)
        CyrkErrorCodes setup()
        cpp_bool check_status()
        void take_step()
        void solve()
        NowStatePointers get_now_state()
        void diffeq()
        CyrkErrorCodes set_cython_extension_instance(
            cpy_ref.PyObject* cython_extension_class_instance,
            DiffeqMethod py_diffeq_method
            )
        void py_diffeq()


# =====================================================================================================================
# Import CySolver Runge-Kutta Integrators
# =====================================================================================================================
cdef extern from "rk.cpp" nogil:

    cdef cppclass RKConfig(ProblemConfig):
        RKConfig(
            DiffeqFuncType diffeq_ptr_,
            double t_start_,
            double t_end_,
            vector[double]& y0_vec_,
            vector[char]& args_vec_,
            vector[double]& t_eval_vec_,
            size_t num_extra_,
            size_t expected_size_,
            size_t max_num_steps_,
            size_t max_ram_MB_,
            PreEvalFunc pre_eval_func_,
            cpp_bool capture_dense_output_,
            cpp_bool force_retain_solver_,
            vector[double]& rtols_,
            vector[double]& atols_,
            double max_step_size_,
            double first_step_size_)
        
        vector[double] rtols
        vector[double] atols
        double max_step_size
        double first_step_size

        void update_properties(
            DiffeqFuncType diffeq_ptr_,
            double t_start_,
            double t_end_,
            vector[double]& y0_vec_,
            vector[char]& args_vec_,
            vector[double]& t_eval_vec_,
            size_t num_extra_,
            size_t expected_size_,
            size_t max_num_steps_,
            size_t max_ram_MB_,
            PreEvalFunc pre_eval_func_,
            cpp_bool capture_dense_output_,
            cpp_bool force_retain_solver_,
            vector[double]& rtols_,
            vector[double]& atols_,
            double max_step_size_,
            double first_step_size_)
        void initialize()
        void update_properties_from_config(RKConfig* new_config_ptr)


    cdef cppclass RKSolver(CySolverBase):
        RKSolver()
        RKSolver(CySolverResult* storage_ptr_)
        void set_Q_order(size_t* Q_order_ptr)
        void set_Q_array(double* Q_ptr)
        CyrkErrorCodes setup()

    cdef cppclass RK23(RKSolver):
        RK23()
        RK23(CySolverResult* storage_ptr_)
    
    cdef cppclass RK45(RKSolver):
        RK45()
        RK45(CySolverResult* storage_ptr_)

    cdef cppclass DOP853(RKSolver):
        DOP853()
        DOP853(CySolverResult* storage_ptr_)

# =====================================================================================================================
# Import the C++ cysolve_ivp helper function
# =====================================================================================================================
cdef extern from "cysolve.cpp" nogil:
    # Pure C++ and Cython implementation

    cdef void baseline_cysolve_ivp_noreturn(
        CySolverResult* solution_ptr,
        DiffeqFuncType diffeq_ptr,
        double t_start,
        double t_end,
        vector[double]& y0_vec,
        size_t expected_size,
        size_t num_extra,
        vector[char]& args_vec,
        size_t max_num_steps,
        size_t max_ram_MB,
        cpp_bool capture_dense_output,
        vector[double]& t_eval_vec,
        PreEvalFunc pre_eval_func,
        vector[double]& rtols,
        vector[double]& atols,
        double max_step_size,
        double first_step_size
        )

    cdef unique_ptr[CySolverResult] baseline_cysolve_ivp(
        DiffeqFuncType diffeq_ptr,
        double t_start,
        double t_end,
        vector[double]& y0_vec,
        ODEMethod integration_method,
        size_t expected_size,
        size_t num_extra,
        vector[char]& args_vec,
        size_t max_num_steps,
        size_t max_ram_MB,
        cpp_bool capture_dense_output,
        vector[double]& t_eval_vec,
        PreEvalFunc pre_eval_func,
        vector[double]& rtols,
        vector[double]& atols,
        double max_step_size,
        double first_step_size
        )


# =====================================================================================================================
# Cython-based wrapper for baseline_cysolve_ivp that carries default values.
# =====================================================================================================================
cdef void cysolve_ivp_noreturn(
    CySolverResult* solution_ptr,
    DiffeqFuncType diffeq_ptr,
    const double t_start,
    const double t_end,
    vector[double] y0_vec,
    double rtol = *,
    double atol = *,
    vector[char] args_vec = *,
    size_t num_extra = *,
    size_t max_num_steps = *,
    size_t max_ram_MB = *,
    bint dense_output = *,
    vector[double] t_eval_vec = *,
    PreEvalFunc pre_eval_func = *,
    vector[double] rtols_vec = *,
    vector[double] atols_vec = *,
    double max_step = *,
    double first_step = *,
    size_t expected_size = *
    ) noexcept nogil

cdef CySolveOutput cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double t_start,
    const double t_end,
    vector[double] y0_vec,
    ODEMethod method = *,
    double rtol = *,
    double atol = *,
    vector[char] args_vec = *,
    size_t num_extra = *,
    size_t max_num_steps = *,
    size_t max_ram_MB = *,
    bint dense_output = *,
    vector[double] t_eval_vec = *,
    PreEvalFunc pre_eval_func = *,
    vector[double] rtols_vec = *,
    vector[double] atols_vec = *,
    double max_step = *,
    double first_step = *,
    size_t expected_size = *
    ) noexcept nogil

cdef CySolveOutput cysolve_ivp_gil(
    DiffeqFuncType diffeq_ptr,
    const double t_start,
    const double t_end,
    vector[double] y0_vec,
    ODEMethod method = *,
    double rtol = *,
    double atol = *,
    vector[char] args_vec = *,
    size_t num_extra = *,
    size_t max_num_steps = *,
    size_t max_ram_MB = *,
    bint dense_output = *,
    vector[double] t_eval_vec = *,
    PreEvalFunc pre_eval_func = *,
    vector[double] rtols_vec = *,
    vector[double] atols_vec = *,
    double max_step = *,
    double first_step = *,
    size_t expected_size = *
    ) noexcept
