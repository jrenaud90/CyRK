# TODO: This document is very sparse. There are currently many ways to interface with CyRK's C++ backend as well as other features and gotchas. This document will be expanded in the future to highlight all of these. In the mean time, please feel free to post questions as GitHub Issues.

_Many of the examples below can be found in the interactive "Demos/Advanced CySolver Examples.ipynb"._

# Arbitrary Additional Arguments

`cysolve_ivp` allows users to specify arbitrary additional arguments which are passed to the differential equation at each evaluation. Many problems require additional arguments and they often take the form of numerical parameters (double precision floating point numbers). However, occasionally more advanced information may need to be passed to the diffeq such as boolean flags, complex numbers, and perhaps even strings or whole other classes or structs. Below we outline how to utilize this feature and any limitations.

Before we describe arbitrary arguments, first we will outline basic argument usage assuming a user has a differential equation that requires an array of floating point numbers.
```cython
%%cython --force 
# distutils: language = c++
# distutils: extra_compile_args = /std:c++20
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sin

from CyRK.cy.cysolverNew cimport (
    cysolve_ivp, find_expected_size, WrapCySolverResult, DiffeqFuncType, MAX_STEP, EPS_100, INF,
    CySolverResult, CySolveOutput
    )

cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr) noexcept nogil:
    # Arguments are passed in as a void pointer to preallocated and instantiated memory. 
    # This memory could be stack or heap allocated. If it is not valid then you will run into seg faults.
    cdef double* args_dbl_ptr = <double*>args_ptr
    cdef double l = args_dbl_ptr[0]
    cdef double m = args_dbl_ptr[1]
    cdef double g = args_dbl_ptr[2]

    cdef double coeff_1 = (-3. * g / (2. * l))
    cdef double coeff_2 = (3. / (m * l**2))

    # Unpack y
    cdef double y0, y1, torque
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    # External torque
    torque = 0.1 * sin(t)

    dy_ptr[0] = y1
    dy_ptr[1] = coeff_1 * sin(y0) + coeff_2 * torque


# Now we define a function that builds our inputs and calls `cysolve_ivp` to solve the problem
def run():
    cdef DiffeqFuncType diffeq = pendulum_diffeq

    # Define time domain
    cdef double[2] time_span_arr = [0.0, 10.0]
    cdef double* t_span_ptr = &time_span_arr[0]
    
    # Define initial conditions
    cdef double[2] y0_arr = [0.01, 0.0]
    cdef double* y0_ptr = &y0_arr[0]
    
    # Define our arguments.
    cdef double[3] args_arr = [1.0, 1.0, 9.81]
    cdef double* args_dbl_ptr = &args_arr[0]

    # To work with cysolve_ivp, we must cast the args ptr to a void pointer
    cdef void* args_ptr = <void*>args_dbl_ptr

    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_span_ptr,
        y0_ptr,
        2,  # Number of dependent variables
        method = 1, # Integration method
        rtol = 1.0e-6,
        atol = 1.0e-8,
        args_ptr = args_ptr
        )

    # If we want to pass the solution back to python we need to wrap it in a CyRK `WrapCySolverResult` class.
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(result)

    return pysafe_result

result_reg = run()
print("\n\nIntegration success =", result_reg.success, "\n\tNumber of adaptive time steps required:", result_reg.size)
print("Integration message:", result_reg.message)
```

Now assume that we have a new diffeq that is similar to before but it allows for atmospheric drag in the system. For the purposes of demonstration, assume the user wants to turn on or off that drag by passing in a boolean flag. The previous example will not work because the arg array is an array of doubles and can not contain other types of data such as bools. Instead the user needs to build a custom data structure and pass the address of that structure to cysolve_ivp. Finally, inside the diffeq function the arg pointer must be cast back to the known data type in order to access the data.

```cython
%%cython --force 
# distutils: language = c++
# distutils: extra_compile_args = /std:c++20
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sin
from libcpp cimport bool as cpp_bool

from CyRK.cy.cysolverNew cimport (
    cysolve_ivp, find_expected_size, WrapCySolverResult, DiffeqFuncType, MAX_STEP, EPS_100, INF,
    CySolverResult, CySolveOutput
    )

cdef struct PendulumArgs:
    # Structure that contains heterogeneous types
    cpp_bool use_drag
    double g
    double l
    double m

cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr) noexcept nogil:
    # Arg pointer still must be listed as a void pointer or it will not work with cysolve_ivp.
    # But now the user can recast that void pointer to the structure they wish.
    cdef PendulumArgs* pendulum_args_ptr = <PendulumArgs*>args_ptr
    # And easily access its members which can be many heterogeneous types.
    cdef double l = pendulum_args_ptr.l
    cdef double m = pendulum_args_ptr.m
    cdef double g = pendulum_args_ptr.g
    cdef double use_drag = pendulum_args_ptr.use_drag

    cdef double coeff_1 = (-3. * g / (2. * l))
    cdef double coeff_2 = (3. / (m * l**2))

    # Unpack y
    cdef double y0, y1, torque
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    # External torque
    torque = 0.1 * sin(t)

    dy_ptr[0] = y1
    dy_ptr[1] = coeff_1 * sin(y0) + coeff_2 * torque

    if use_drag:
        dy_ptr[1] -= 1.5 * y1

# Now we define a function that builds our inputs and calls `cysolve_ivp` to solve the problem
def run():
    cdef DiffeqFuncType diffeq = pendulum_diffeq

    # Define time domain
    cdef double[2] time_span_arr = [0.0, 10.0]
    cdef double* t_span_ptr = &time_span_arr[0]
    
    # Define initial conditions
    cdef double[2] y0_arr = [0.01, 0.0]
    cdef double* y0_ptr = &y0_arr[0]
    
    # Define our arguments.
    # We now have a a structure that we need to allocate memory for.
    # For this example, let's do it on the stack. 
    cdef PendulumArgs pendulum_args = PendulumArgs(True, 9.81, 1.0, 1.0)
    # We need to pass in a void pointer to cysolve_ivp, so let's cast the address of the struct to void*
    cdef void* args_ptr = <void*>&pendulum_args

    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_span_ptr,
        y0_ptr,
        2,  # Number of dependent variables
        method = 1, # Integration method
        rtol = 1.0e-6,
        atol = 1.0e-8,
        args_ptr = args_ptr
        )

    # If we want to pass the solution back to python we need to wrap it in a CyRK `WrapCySolverResult` class.
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(result)

    return pysafe_result

result_drag = run()
print("\n\nIntegration success =", result_drag.success, "\n\tNumber of adaptive time steps required:", result_drag.size)
print("Integration message:", result_drag.message)
```

# Pre-Evaluation Function

It is occasionally advantageous for users to define differential equation functions that utilize a "pre-evaluation" function that take in the current state and provide parameters which are then used to find dydt. This allows different models to be defined as modified versions of this pre-eval function without changing the rest of the differential equation.

Consider the example of a pendulum that is experiencing drag and one that is not. The equations of motion for the pendulum are only slightly modified between the two cases. Rather than writing a different differential equation we can instead write one diffeq and two different pre-evaluation functions. 

# TODO: Left off: change demo example to have two pre-eval funcs that match the above description. finish this section. 
