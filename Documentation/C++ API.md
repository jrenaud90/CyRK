# CyRK's C++ API Description
CyRK uses a C++ backend for most of its functionality. These files are described below.

This backend can be found in a dedicated repository [here](https://github.com/jrenaud90/CyRK_CPP).

## "CyRK/cy/common.hpp(cpp)"
Contains common functions and global constants.

## "CyRK/cy/cysolution.hpp(cpp)"
Contains the `CySolverResult` class definition. The purpose of this structure is to store the results of CyRK's CySolver
integrator. It stores data in C++ vectors which are dynamically resized if required. The CySolverResult class must be
instantiated (using a shared smart pointer) and passed to CySolver functions and class instances.

## "CyRK/cy/cysolver.hpp(cpp)"
Contains the `CySolverBase` class definition. This is the main building block for CyRK's integrators. It contains
state variables that track the progress and current state of the integration. Methods to perform integration steps or
a complete integration. Methods to provide hooks so that Python functions can be called from within C++
(see "PySolver API.md" for more details). The CySolver class will automatically save data to a pre-initialized 
`CySolverResult` instance with each successful integration step. It also has the responsibility of updating error or
info messages to the CySolverResult instance.

## "CyRK/cy/cysolve.hpp(cpp)"
Contains helper functions that handle a lot of the boilerplate initialization of classes and memory allocation in a
user-friendly format. These helper functions are provided for both pure C++ (which is later wrapped by Cython) and
a version that works with Python functions.

```C++
// Pure C++ version "cysolve_ivp"
baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const size_t num_y,
    const int method,
    // General optional arguments
    const size_t expected_size = 0,
    const size_t num_extra = 0,
    const double* args_ptr = nullptr,
    // rk optional arguments
    const size_t max_num_steps = 0,
    const size_t max_ram_MB = 2000,
    const double rtol = 1.0e-3,
    const double atol = 1.0e-6,
    const double* rtols_ptr = nullptr,
    const double* atols_ptr = nullptr,
    const double max_step_size = MAX_STEP,
    const double first_step_size = 0.0
)


// Version that allows for Python functions
class PySolver
{
    PySolver(
        int integration_method,
        // Cython class instance used for pyhook
        PyObject* cython_extension_class_instance,
        DiffeqMethod cython_extension_class_diffeq_method,
        // Regular integrator inputs
        std::shared_ptr<CySolverResult> solution_ptr,
        const double t_start,
        const double t_end,
        const double* y0_ptr,
        const size_t num_y,
        // General optional arguments
        const size_t num_extra,
        const double* args_ptr,
        // rk optional arguments
        const size_t max_num_steps,
        const size_t max_ram_MB,
        const double rtol,
        const double atol,
        const double* rtols_ptr,
        const double* atols_ptr,
        const double max_step_size,
        const double first_step_size
    );
}
```

## "CyRK/cy/rk.hpp(cpp)"
Provides classes that wrap `CySolverBase` and provide Runge-Kutta integration methods and constants. Each integrator has
a unique integer used to select it via `integration_method` in various function calls. These integers can be imported
via `[Capital Method Name]_METHOD_INT` (e.g., RK23 would be found with "RK23_METHOD_INT") found in "rk.hpp".

Currently available functions and associated integration method integer:
- RK23 : 0
    - Explicit Runge-Kutta method of order 3 (error control of order 2)
- RK45 : 1 
    - Explicit Runge-Kutta method of order 5 (error control of order 4)
- DOP853 : 2
    - Explicit Runge-Kutta method of order 8 (error control of combination of order 5 and 3)
