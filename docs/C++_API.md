# CyRK's C++ API Description
CyRK uses a C++ backend for most of its functionality. These files are described below.

This backend can be found in a dedicated repository [here](https://github.com/jrenaud90/CyRK_CPP).

## "common.hpp(cpp)"
Contains common functions and global constants.

## "cysolution.hpp(cpp)"
Contains the `CySolverResult` class definition. The purpose of this structure is to store the results of CyRK's CySolver
integrator.

It also stores the CySolver itself and any required data the solver needs to perform calculations.
 
Integration results are stored as C++ vectors which are dynamically resized if required. The CySolverResult class
must be instantiated (using a shared smart pointer) and passed to CySolver functions and class instances.

## "cysolver.hpp(cpp)"
Contains the `CySolverBase` class definition. This is the main building block for CyRK's integrators. It contains
state variables that track the progress and current state of the integration. Methods to perform integration steps or
a complete integration. Methods to provide hooks so that Python functions can be called from within C++
(see "PySolver API.md" for more details). The CySolver class will automatically save data to its parent
`CySolverResult` instance with each successful integration step. It also has the responsibility of updating error or
info messages to the CySolverResult instance.

## "cysolve.hpp(cpp)"
Contains helper functions that handle a lot of the boilerplate initialization of classes and memory allocation in a
user-friendly format.

```C++
// Pure C++ version "cysolve_ivp"
void baseline_cysolve_ivp_noreturn(
        CySolverResult* solution_ptr,
        DiffeqFuncType diffeq_ptr,
        double t_start,
        double t_end,
        std::vector<double> y0_vec,
        // General optional arguments
        size_t> expected_size,
        size_t> num_extra,
        std::vector<char> args_vec,
        size_t> max_num_steps,
        size_t> max_ram_MB,
        bool> capture_dense_output,
        std::vector<double> t_eval_vec,
        PreEvalFunc> pre_eval_func,
        // rk optional arguments
        std::vector<double> rtols,
        std::vector<double> atols,
        double max_step_size,
        double first_step_size
    )

std::unique_ptr<CySolverResult> baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    double t_start,
    double t_end,
    std::vector<double> y0_vec,
    ODEMethod integration_method,
    // General optional arguments
    size_t> expected_size,
    size_t> num_extra,
    std::vector<char> args_vec,
    size_t> max_num_steps,
    size_t> max_ram_MB,
    bool> capture_dense_output,
    std::vector<double> t_eval_vec,
    PreEvalFunc> pre_eval_func,
    // rk optional arguments
    std::vector<double> rtols,
    std::vector<double> atols,
    double max_step_size,
    double first_step_size
)
```

## "CyRK/cy/rk.hpp(cpp)"
Provides classes that wrap `CySolverBase` and provide Runge-Kutta integration methods and constants. Each integrator has
a unique integer used to select it via `integration_method` in various function calls. These integers are defined in 
an enum class `ODEMethod` which can be python imported or cython cimported `from CyRK import ODEMethod; ODEMethod.RK45`.

Currently available functions and associated integration method integer:
- RK23 : ODEMethod.RK23
    - Explicit Runge-Kutta method of order 3 (error control of order 2)
- RK45 : ODEMethod.RK45
    - Explicit Runge-Kutta method of order 5 (error control of order 4)
- DOP853 : ODEMethod.DOP853
    - Explicit Runge-Kutta method of order 8 (error control of combination of order 5 and 3)
