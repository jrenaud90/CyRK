from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string
from libcpp.map cimport map as cpp_map

# =====================================================================================================================
# Import common functions and constants
# =====================================================================================================================
cdef extern from "c_common.cpp" nogil:
    cpdef enum class CyrkErrorCodes(int):
        CONVERGED,
        INITIALIZING,
        EVENT_TERMINATED,
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
        EVENT_SETUP_FAILED,
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
