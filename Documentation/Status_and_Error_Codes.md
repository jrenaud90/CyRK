# Error and Status Codes

All of the solvers have an internal status code that is updated as integration is performed.

These codes and their accompanying messages are defined in `CyRK.cy.c_common.hpp`. 

## Status Codes and Messages

Below are a list of status code ints followed by their messages. This was last updated for CyRK v0.17.0.

```C++

enum class CyrkErrorCodes : int {
    
    // Temporary statuses
    CONVERGED = 20,
    INITIALIZING = 10,

    // "Success" statuses
    EVENT_TERMINATED = 2,
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

    // Problem with CyRK.optimize start at -30
    OPTIMIZE_SIGN_ERROR = -30,
    OPTIMIZE_CONVERGENCE_ERROR = -31,

    // Memory allocation starts at -40
    MEMORY_ALLOCATION_ERROR = -40,
    VECTOR_SIZE_EXCEEDS_LIMITS = -41,

    // Problems with integration start at -50
    NUMBER_OF_EQUATIONS_IS_ZERO = -50,
    MAX_ITERATIONS_HIT = -51,
    MAX_STEPS_USER_EXCEEDED = -52,
    MAX_STEPS_SYSARCH_EXCEEDED = -53,
    STEP_SIZE_ERROR_SPACING = -54,
    STEP_SIZE_ERROR_ACCEPTANCE = -55,
    DENSE_BUILD_FAILED = -56,
    INTEGRATION_NOT_SUCCESSFUL = -57,

    // Problems related to events start at -60
    EVENT_SETUP_FAILED = -60,

    // Python related problems start at -70
    ERROR_IMPORTING_PYTHON_MODULE = -70,

    // RK-specific issues start at -80
    BAD_INITIAL_STEP_SIZE = -80,

    OTHER_ERROR = -99,
    UNSET_ERROR_CODE = -100
};

inline const std::map<CyrkErrorCodes, std::string> CyrkErrorMessages = {
    { CyrkErrorCodes::CONVERGED,
      "An optimization routine has successfully converged." },

    { CyrkErrorCodes::INITIALIZING,
      "Initializing. If you see this message then it was likely interrupted." },
    
    { CyrkErrorCodes::EVENT_TERMINATED,
      "Integration ended early: An event has been triggered the maximum allowed times. No issues detected." },

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
```
