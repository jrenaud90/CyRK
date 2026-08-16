# CyRK's C++ API Description
CyRK uses a C++ backend for most of its functionality. Most of these files can be found in "CyRK/cy" and
are described below.

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
        std::vector<Event> events_vec,
        // Error control arguments
        std::vector<double> rtols,
        std::vector<double> atols,
        double max_step_size,
        double first_step_size,
        // Memory management
        bool force_retain_solver,
        // Only used by the implicit methods; null means estimate the Jacobian by finite differences
        JacobianFuncType jac_ptr
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
    std::vector<Event> events_vec,
    // Error control arguments
    std::vector<double> rtols,
    std::vector<double> atols,
    double max_step_size,
    double first_step_size,
    // Memory management
    bool force_retain_solver,
    // Only used by the implicit methods; null means estimate the Jacobian by finite differences
    JacobianFuncType jac_ptr
)
```

## "rk.hpp(cpp)"
Provides classes that wrap `CySolverBase` and provide Runge-Kutta integration methods and constants. Each integrator has
a unique integer used to select it via `integration_method` in various function calls. These integers are defined in 
an enum class `ODEMethod` which can be python imported or cython cimported `from CyRK import ODEMethod; ODEMethod.RK45`.
New methods are always appended to the end of that enum so that the existing integer values stay stable.

Currently available functions and associated integration method integer:
- RK23 : ODEMethod.RK23
    - Explicit Runge-Kutta method of order 3 (error control of order 2)
- RK45 : ODEMethod.RK45
    - Explicit Runge-Kutta method of order 5 (error control of order 4)
- DOP853 : ODEMethod.DOP853
    - Explicit Runge-Kutta method of order 8 (error control of combination of order 5 and 3)
- BDF : ODEMethod.BDF
    - Implicit multi-step method based on backward differentiation formulas of order 1 to 5
- LSODA : ODEMethod.LSODA
    - Adams / BDF method with automatic stiffness detection and switching

## "bdf.hpp(cpp)"
Provides the `BDF` class, an implicit multi-step integrator built on the backward differentiation formulas. It needs no
configuration beyond what `ProblemConfig` already carries.

## "lsoda.hpp(cpp)" and "c_lsoda.hpp(cpp)"
`lsoda.hpp` provides the `LSODA` class along with `LSODAConfig`, which adds the options that only LSODA understands
(`min_step_size`, `max_order_nonstiff`, `max_order_stiff`, `num_lower`, and `num_upper`). `CySolverResult` builds an
`LSODAConfig` automatically when `ODEMethod::LSODA` is selected, so retrieve it with a `dynamic_cast` before setting
those options.

`c_lsoda.hpp(cpp)` holds the vendored ODEPACK implementation that the `LSODA` class drives one step at a time. It is
third-party code; see the `Third-Party Code` section of the license for its notices.

## "c_lu.hpp(cpp)"
Dense and banded LU factorization routines (the equivalents of LAPACK's dgetrf, dgetrs, dgbtrf, and dgbtrs) that the
implicit methods use to factorize their iteration matrices. They are implemented here so that CyRK does not have to
link against an external BLAS or LAPACK library. All matrices use LAPACK's column-major storage.

## Memory Usage
The following formulas approximate the total memory footprint in bytes of the underlying C++ structures. Only think of these as estimates. All values are in kB. 
These assume that extra outputs, dense outputs, and events are all off.

$S = $ Solution Size.

$N = $ Number of dependent y's (for this we assume num_dy = num_y).

### RK23
$$8S(N+1)+120N+1,528$$
### RK45
$$8S(N+1)+144N+1,528$$
### DOP853
$$8S(N+1)+232N+1,528$$
### BDF
$$8S(N+1)+16N^2+240N+1,528$$
The $N^2$ term is the Jacobian plus its LU factorization, both stored as dense matrices.
### LSODA (dense Jacobian)
$$8S(N+1)+8N^2+140N+3,600$$
### LSODA (banded Jacobian)
$$8S(N+1)+8N(2 \cdot lband + uband + 10)+68N+3,600$$

Note the $N^2$ term for the implicit methods. CyRK checks that footprint against `max_ram_MB` during setup and
refuses to start rather than attempting a solve that could not finish in reasonable time. Narrowing LSODA's Jacobian
to a band brings the cost back down to linear in $N$.