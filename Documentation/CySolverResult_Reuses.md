# CySolver & PySolver Reuses
_See the demos in "2 - Advanced Examples" jupyter notebook._

CyRK's cysolve_ivp and pysolve_ivp methods require allocating various chunks of memory that are required to solve the ODE.
Depending on your needs you may need to rerun a ODE many many times with only minor changes. For example slightly different 
initial conditions. If your ODE does not require many integration steps (particularly true for small time spans) then the 
allocation of memory for the storage can start to become a significant fraction of the time. To mitigate this CyRK provides
ways to reuse result storage to avoid allocations (or have minor reallocations). The examples below show you how you can 
make use of this system.

**Warning! Reusing a result structure will erase all previously stored data.**

## PySolver Example
```python
# Note if using this format, `dy` must be the first argument. Additionally, a special flag must be set to True when calling pysolve_ivp, see below.
def cy_diffeq(dy, t, y):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]

# Since this is pure python we can njit it safely
from numba import njit
cy_diffeq = njit(cy_diffeq)
    
import numpy as np
from CyRK import pysolve_ivp

initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8

result = \
    pysolve_ivp(cy_diffeq, time_span, initial_conds,
                method="RK45", rtol=rtol, atol=atol, pass_dy_as_arg=True)

# Use the result's memory allocation to re-call pysolve_ivp
result = \
    pysolve_ivp(cy_diffeq, time_span, initial_conds,
                method="RK45", rtol=rtol, atol=atol, pass_dy_as_arg=True,
                solution_reuse=result) # Set the reuse argument.

print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(result.t, result.y[0], c='r')
ax.plot(result.t, result.y[1], c='b')
plt.show()

# We can change parameters too
new_initial_conds = np.asarray((3., 0.1), dtype=np.float64, order='C')
result = \
    pysolve_ivp(cy_diffeq, time_span, new_initial_conds,
                method="RK45", rtol=rtol, atol=atol, pass_dy_as_arg=True,
                solution_reuse=result) # Set the reuse argument.

print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(result.t, result.y[0], c='r')
ax.plot(result.t, result.y[1], c='b')
plt.show()
```

## CySolver Example
```cython
%%cython --force
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from libcpp.utility cimport move
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
np.import_array()

from CyRK cimport cysolve_ivp, cysolve_ivp_noreturn, DiffeqFuncType, WrapCySolverResult, CySolveOutput, PreEvalFunc

cdef void cython_diffeq(double* dy, double t, double* y, char* args, PreEvalFunc pre_eval_func) noexcept nogil:

    # Build Coeffs
    cdef double coeff_1 = (1. - 0.01 * y[1])
    cdef double coeff_2 = (0.02 * y[0] - 1.)
    
    # Store results
    dy[0] = coeff_1 * y[0]
    dy[1] = coeff_2 * y[1]

# Import the required functions from CyRK
from CyRK cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput

# Let's get the integration number for the RK45 method
from CyRK cimport ODEMethod

def run_reuse_cysolver(tuple t_span, double[::1] y0):
    
    # Cast our diffeq to the accepted format
    cdef DiffeqFuncType diffeq = cython_diffeq
    
    # Convert the python user input to pure C types
    cdef size_t num_y          = len(y0)
    cdef double t_start        = t_span[0]
    cdef double t_end          = t_span[1]
    cdef vector[double] y0_vec = vector[double](num_y)
    cdef size_t yi
    for yi in range(num_y):
        y0_vec[yi] = y0[yi]

    # Run the integrator!
    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_start,
        t_end,
        y0_vec,
        method = ODEMethod.RK45,
        rtol = 1.0e-7,
        atol = 1.0e-8
    )

    # Run again using the noreturn version so we can reuse the integration storage.
    cysolve_ivp_noreturn(
        result.get(),  # result is a unique pointer; we need to pass the raw pointer to the baseline_cysolve_ivp
        diffeq,
        t_start,
        t_end,
        y0_vec,
        rtol = 1.0e-7,
        atol = 1.0e-8
    )

    # Like with PySolver, we can change the inputs 
    cdef double t_end_2 = 10.0 * t_end
    cysolve_ivp_noreturn(
        result.get(),  # result is a unique pointer; we need to pass the raw pointer to the baseline_cysolve_ivp
        diffeq,
        t_start,
        t_end_2,
        y0_vec,
        rtol = 1.0e-7,
        atol = 1.0e-8
    )

    # The CySolveOutput is not accesible via Python. We need to wrap it first
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(move(result))

    return pysafe_result

```