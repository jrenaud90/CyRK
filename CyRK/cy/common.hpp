#pragma once

#include <map>
#include <string>
#include <cmath>
#include <limits>

// Pre-processor constants
static const size_t BUFFER_SIZE     = 16;
static const size_t PRE_ALLOC_STEPS = 256;
static const size_t PRE_ALLOC_NUMY  = 16;

enum class CyrkErrorCodes : int {
    INITIALIZING = 10,
    SUCCESSFUL_INTEGRATION = 1,
    NO_ERROR = 0,

    // Error status
    GENERAL_ERROR = -1,
    PROPERTY_NOT_SET = -2,
    UNSUPPORTED_UNKNOWN_MODEL = -3,
    UNINITIALIZED_CLASS = -4,
    CYSOLVER_INITIALIZATION_ERROR = -5,
    INCOMPATIBLE_INPUT = -6,
    ATTRIBUTE_ERROR = -7,
    BOUNDS_ERROR = -8,
    ARGUMENT_NOT_SET = -9,
    ARGUMENT_ERROR = -10,

    // Problems with Setup start at -20
    SETUP_NOT_CALLED = -20,
    DENSE_OUTPUT_NOT_SAVED = -21,
    BAD_CONFIG_DATA = -22,

    // Problem with vectors start at -30
    VECTOR_SIZE_EXCEEDS_LIMITS = -30,

    // Memory allocation starts at -40
    MEMORY_ALLOCATION_ERROR = -40,

    // Problems with integration start at -50
    NUMBER_OF_EQUATIONS_IS_ZERO = -50,
    MAX_ITERATIONS_HIT = -51,
    MAX_STEPS_USER_EXCEEDED = -52,
    MAX_STEPS_SYSARCH_EXCEEDED = -53,
    STEP_SIZE_ERROR_SPACING = -54,
    STEP_SIZE_ERROR_ACCEPTANCE = -55,
    DENSE_BUILD_FAILED = -56,
    INTEGRATION_NOT_SUCCESSFUL = -57,

    // Python related problems start at -70
    ERROR_IMPORTING_PYTHON_MODULE = -70,

    // RK-specific issues start at -80
    BAD_INITIAL_STEP_SIZE = -80,

    OTHER_ERROR = -99,
    UNSET_ERROR_CODE = -100
};

inline const std::map<CyrkErrorCodes, std::string> CyrkErrorMessages = {
    { CyrkErrorCodes::INITIALIZING,
      "Initializing. If you see this message then it was likely interrupted." },

    { CyrkErrorCodes::SUCCESSFUL_INTEGRATION,
      "Integration completed without issue." },

    { CyrkErrorCodes::NO_ERROR,
      "No errors were encountered." },

    { CyrkErrorCodes::GENERAL_ERROR,
      "An unspecified error has occurred." },

    { CyrkErrorCodes::PROPERTY_NOT_SET,
      "A required property was not set." },

    { CyrkErrorCodes::UNSUPPORTED_UNKNOWN_MODEL,
      "The provided model is currently unsupported or not known." },

    { CyrkErrorCodes::UNINITIALIZED_CLASS,
      "A class object was not fully initialized." },

    { CyrkErrorCodes::CYSOLVER_INITIALIZATION_ERROR,
      "Error in CySolver initialization." },

    { CyrkErrorCodes::INCOMPATIBLE_INPUT,
      "The provided input is incompatible with other input or the current object(s) state." },

    { CyrkErrorCodes::ATTRIBUTE_ERROR,
      "Error when accessing or modifying one or more attributes/properties." },

    { CyrkErrorCodes::BOUNDS_ERROR,
      "A bounds error was encountered." },

    { CyrkErrorCodes::ARGUMENT_NOT_SET,
      "A required function or method argument was not set." },

    { CyrkErrorCodes::SETUP_NOT_CALLED,
      "An object's additional setup function was not called before its methods were used." },

    { CyrkErrorCodes::DENSE_OUTPUT_NOT_SAVED,
      "Can't call solution: Dense output was not saved." },

    { CyrkErrorCodes::BAD_CONFIG_DATA,
      "Error during setup: Provided configuration data does not make sense or is missing required parameters." },

    { CyrkErrorCodes::VECTOR_SIZE_EXCEEDS_LIMITS,
      "A C++ vector object's size exceeds limits imposed by user or architecture." },

    { CyrkErrorCodes::MEMORY_ALLOCATION_ERROR,
      "There was an error while allocating memory." },

    { CyrkErrorCodes::NUMBER_OF_EQUATIONS_IS_ZERO,
      "The number of dependent `y` values is 0. There are no differential equations to solve." },

    { CyrkErrorCodes::MAX_ITERATIONS_HIT,
      "The maximum number of iterations was hit." },

    { CyrkErrorCodes::MAX_STEPS_USER_EXCEEDED,
      "Maximum number of steps (set by user) exceeded during integration." },

    { CyrkErrorCodes::MAX_STEPS_SYSARCH_EXCEEDED,
      "Maximum number of steps (set by system architecture) exceeded during integration." },

    { CyrkErrorCodes::STEP_SIZE_ERROR_SPACING,
      "Error in step size calculation: Required step size is less than spacing between numbers." },

    { CyrkErrorCodes::STEP_SIZE_ERROR_ACCEPTANCE,
      "Error in step size calculation: Error in step size acceptance." },

    { CyrkErrorCodes::DENSE_BUILD_FAILED,
      "Error during dense output build." },

    { CyrkErrorCodes::INTEGRATION_NOT_SUCCESSFUL,
      "Can't call solution: Integration has not been completed or it was unsuccessful." },

    { CyrkErrorCodes::ERROR_IMPORTING_PYTHON_MODULE,
      "There was an error in the C++ backend when trying to import the required Python module." },

    { CyrkErrorCodes::BAD_INITIAL_STEP_SIZE,
      "User-provided initial step size must be a positive number." },

    { CyrkErrorCodes::OTHER_ERROR,
      "An unknown error occurred." },

    { CyrkErrorCodes::UNSET_ERROR_CODE,
      "The error code was never set." }
};


// Integration Constants
// Multiply steps computed from asymptotic behaviour of errors by this.
static const double SAFETY             = 0.9;   // Error coefficient factor (1 == use calculated error; < 1 means be conservative).
static const double MIN_FACTOR         = 0.2;   // Minimum allowed decrease in a step size.
static const double MAX_FACTOR         = 10.0;  // Maximum allowed increase in a step size.
static constexpr double INF            = std::numeric_limits<double>::infinity();
static const double MAX_STEP           = INF;
static constexpr double EPS            = std::numeric_limits<double>::epsilon();
static const double EPS_10             = EPS * 10.0;
static const double EPS_100            = EPS * 100.0;
static constexpr size_t MAX_SIZET_SIZE = std::numeric_limits<size_t>::max();
static constexpr size_t MAX_INT_SIZE   = std::numeric_limits<int>::max();


// Memory management constants
// Assume that a cpu has a L1 of 300KB.Say that this progam will have access to 75 % of that total.
static constexpr double CPU_CACHE_SIZE = 0.75 * 300000.0;
// Number of entities we can fit into that size is based on the size of double(or double complex)
static const double MAX_ARRAY_PREALLOCATE_SIZE_DBL = 600000.0;
static const double MIN_ARRAY_PREALLOCATE_SIZE     = 10.0;
static const double ARRAY_PREALLOC_TABS_SCALE      = 1000.0;  // A delta_t_abs higher than this value will start to grow array size.
static const double ARRAY_PREALLOC_RTOL_SCALE      = 1.0e-5;  // A rtol lower than this value will start to grow array size.

// Solution parameters
static const double DYNAMIC_GROWTH_RATE = 1.618;
// To learn why the golden ratio is used, read this:
// https://stackoverflow.com/questions/1100311/what-is-the-ideal-growth-rate-for-a-dynamically-allocated-array
static constexpr double SIZE_MAX_DBL = 0.99 * SIZE_MAX;



typedef void (*PreEvalFunc)(char*, double, double*, char*);

typedef void (*DiffeqFuncType)(double*, double, double*, char*, PreEvalFunc);

struct MaxNumStepsOutput
{
    bool user_provided_max_num_steps;
    size_t max_num_steps;

    MaxNumStepsOutput(bool user_provided, size_t max_steps) :
        user_provided_max_num_steps(user_provided),
        max_num_steps(max_steps) { };
};


void round_to_2(size_t& initial_value) noexcept;

MaxNumStepsOutput find_max_num_steps(
    const size_t num_y,
    const size_t num_extra,
    const size_t max_num_steps,
    const size_t max_ram_MB);


size_t find_expected_size(
    size_t num_y,
    size_t num_extra,
    double t_delta_abs,
    double rtol_min);
