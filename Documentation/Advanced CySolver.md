# Advanced CySolver 
_See the demos in "2 - Advanced Examples" jupyter notebook._
_TODO: This document needs expansion. There are currently many ways to interface with CyRK's C++ backend as well as other features and gotchas. This document will be expanded in the future to highlight all of these. In the mean time, please feel free to post questions as GitHub Issues._

_Many of the examples below can be found in the interactive "Demos/Advanced CySolver Examples.ipynb" jupyter notebook._

## Arbitrary Additional Arguments

`cysolve_ivp` allows users to specify arbitrary additional arguments which are passed to the differential equation function at each evaluation. Many problems require additional arguments that often take the form of numerical parameters (e.g., double precision floating point numbers). However, occasionally more advanced information may be required such as: boolean flags, complex numbers, and perhaps even strings or whole other classes and structs. Below we outline how to utilize arbitrary types in CySolver's additional arguments and discuss any limitations.

Before we describe arbitrary arguments, first we will outline basic argument usage assuming a user has a differential equation that requires an array of floating point numbers.

**Important Note About Memory Ownership**: CySolverBase makes a copy of the the additional argument structure. If this argument structure contains pointers to other structures or data then it will make a copy of those raw pointers _but does not take ownership of their memory_. If those pointers are later released or changed then CySolver will not know about the change and will have hanging pointers. _We highly recommended against using raw pointers in the additional argument structure._ If you do use them then you need to guarantee that the data they point to stays alive, unchanged, and in the same location throughout the life of the CySolver class.

_Note: This assumes you are working in a Jupyter notebook. If you are not then you can exclude the "%%cython --force" at the beginning of each cell._

```cython
%%cython --force 
# distutils: language = c++
# distutils: extra_compile_args = /std:c++20
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sin
from libcpp.utility cimport move
from libcpp.vector cimport vector

from CyRK cimport cysolve_ivp, WrapCySolverResult, DiffeqFuncType, MAX_STEP, CySolveOutput, PreEvalFunc, ODEMethod

cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Arguments are passed in as char pointers.
    # These point to preallocated and instantiated memory in the CySolver class that is filled with user-provided data.

    # The user can then recast the generic char pointers back to original format of args_ptr. In this case, an array of doubles. 
    # Seg faults will occur if this function recasts to the incorrect type.
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
    cdef double t_start = 0.0
    cdef double t_end = 10.0
    
    # Define initial conditions
    cdef vector[double] y0_vec = vector[double](2)
    y0_vec[0] = 0.01
    y0_vec[1] = 0.0
    
    # Define our arguments in a char vector
    # The size of the vector is the size of the underlying object(s) x number of them.
    cdef vector[char] args_vec = vector[char](sizeof(double) * 3)
    # To make it easier to populate the args_vec we recast it as a double pointer.
    cdef double* args_dbl_ptr = <double*>args_vec.data()
    args_dbl_ptr[0] = 1.0
    args_dbl_ptr[1] = 1.0
    args_dbl_ptr[2] = 9.81

    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_start,
        t_end,
        y0_vec,
        method = ODEMethod.RK45, # Integration method
        rtol = 1.0e-6,
        atol = 1.0e-8,
        args_vec = args_vec
        )

    # If we want to pass the solution back to python we need to wrap it in a CyRK `WrapCySolverResult` class.
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(move(result))

    return pysafe_result

result_reg = run()
print("\n\nIntegration success =", result_reg.success, "\n\tNumber of adaptive time steps required:", result_reg.size)
print("Integration message:", result_reg.message)
```

Now assume that we have a new diffeq that is similar to before but it allows for atmospheric drag in the system. For the purposes of demonstration, assume the user wants to turn on or off that drag by passing in a boolean flag. The previous example will not work because the `args_arr` is an array of doubles and can not contain other types of data such as booleans. Instead the user will need to build a custom data structure and pass the address of that structure to `cysolve_ivp`. Finally, inside the diffeq function the arg pointer must be cast back to that known data type in order to access the data.

```cython
%%cython --force 
# distutils: extra_compile_args = /std:c++20
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
np.import_array()

from libc.string cimport memcpy
from libc.math cimport sin
from libcpp cimport bool as cpp_bool
from libcpp.utility cimport move
from libcpp.vector cimport vector

from CyRK cimport cysolve_ivp, WrapCySolverResult, DiffeqFuncType,MAX_STEP, CySolveOutput, PreEvalFunc, ODEMethod

cdef struct PendulumArgs:
    # Structure that contains heterogeneous types
    cpp_bool use_drag
    double g
    double l
    double m

cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Arg pointer still must be listed as a char pointer or it will not work with cysolve_ivp.
    # But now the user can recast that char pointer to the structure they wish.
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
    cdef double t_start = 0.0
    cdef double t_end = 10.0
    
    # Define initial conditions
    cdef vector[double] y0_vec = vector[double](2)
    y0_vec[0] = 0.01
    y0_vec[1] = 0.0
    
    # Define our arguments.
    # We now have a a structure that we need to allocate memory for.
    # For this example, let's do it on the stack. 
    cdef PendulumArgs pendulum_args = PendulumArgs(True, 9.81, 1.0, 1.0)
    # We need to pass in a char vector to cysolve_ivp, so let's create a vector of the correct size
    cdef vector[char] args_vec = vector[char](sizeof(PendulumArgs))
    # Then we need to copy over the contents to this vector.
    memcpy(args_vec.data(), &pendulum_args, sizeof(PendulumArgs))

    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_start,
        t_end,
        y0_vec,
        method = ODEMethod.RK45,
        rtol = 1.0e-6,
        atol = 1.0e-8,
        args_vec = args_vec
        )

    # If we want to pass the solution back to python we need to wrap it in a CyRK `WrapCySolverResult` class.
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(move(result))

    return pysafe_result

result_drag = run()
print("\n\nIntegration success =", result_drag.success, "\n\tNumber of adaptive time steps required:", result_drag.size)
print("Integration message:", result_drag.message)
```

# Pre-Evaluation Functions

It is occasionally advantageous for users to define differential equation functions that utilize a "pre-evaluation" function that will use the current state to perform calculations that are then used by the diffeq function to find dydt. While this functionality could be hard coded into the diffeq, having a pre-eval function allows for different models to be defined without changing the rest of the differential equation. A large set of different pre-eval functions could then be passed in to subsequent runs of the solver to compare various models. 

Consider the previous example of a pendulum that is experiencing drag versus one that is not. The equations of motion for the pendulum are only slightly modified between the two cases. Rather than writing a different differential equation, or build an additional argument structure as we used above, we can instead write one diffeq and two different pre-evaluation functions. 

```cython
%%cython --force 
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sin
from libcpp.utility cimport move
from libcpp.vector cimport vector

from CyRK cimport cysolve_ivp, WrapCySolverResult, DiffeqFuncType, MAX_STEP, CySolveOutput, PreEvalFunc, ODEMethod

cdef void pendulum_preeval_nodrag(char* output_ptr, double time, double* y_ptr, char* args_ptr) noexcept nogil:
    # Unpack args (in this example we do not need these but they follow the same rules as the Arbitrary Args section discussed above)
    cdef double* args_dbl_ptr = <double*>args_ptr

    # External torque
    cdef double torque = 0.1 * sin(time)

    # Convert output pointer to double pointer so we can store data
    cdef double* output_dbl_ptr = <double*>output_ptr
    output_dbl_ptr[0] = torque
    output_dbl_ptr[1] = 0.0  # No Drag

cdef void pendulum_preeval_withdrag(char* output_ptr, double time, double* y_ptr, char* args_ptr) noexcept nogil:
    # Unpack args (in this example we do not need these but they follow the same rules as the Arbitrary Args section discussed above)
    cdef double* args_dbl_ptr = <double*>args_ptr

    # External torque
    cdef double torque = 0.1 * sin(time)

    # Convert output pointer to double pointer so we can store data
    cdef double* output_dbl_ptr = <double*>output_ptr
    output_dbl_ptr[0] = torque
    output_dbl_ptr[1] = -1.5 * y_ptr[1]  # With Drag


cdef void pendulum_preeval_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:

    # Build other parameters that do not depend on the pre-eval func
    cdef double* args_dbl_ptr = <double*>args_ptr
    cdef double l = args_dbl_ptr[0]
    cdef double m = args_dbl_ptr[1]
    cdef double g = args_dbl_ptr[2]

    cdef double coeff_1 = (-3. * g / (2. * l))
    cdef double coeff_2 = (3. / (m * l**2))

    # Make stack allocated storage for pre eval output
    cdef double[4] pre_eval_storage
    cdef double* pre_eval_storage_ptr = &pre_eval_storage[0]

    # Cast storage to char so we can call function
    cdef char* pre_eval_storage_char_ptr = <char*>pre_eval_storage_ptr

    # Call Pre-Eval Function
    # Note that even though CyRK calls this function a "pre-eval" function, it can be placed anywhere inside the diffeq function. 
    pre_eval_func(pre_eval_storage_char_ptr, t, y_ptr, args_ptr)

    cdef double y0 = y_ptr[0]
    cdef double y1 = y_ptr[1]

    # Use results of pre-eval function to update dy. Note that we are using the double* not the char* here.
    # Even though pre_eval_func was passed the char* it updated the memory that the double* pointed to so we can use it below.
    dy_ptr[0] = y1
    dy_ptr[1] = coeff_1 * sin(y0) + coeff_2 * pre_eval_storage_ptr[0] + pre_eval_storage_ptr[1]


# Now we define a function that builds our inputs and calls `cysolve_ivp` to solve the problem
def run():
    cdef DiffeqFuncType diffeq = pendulum_preeval_diffeq
    
    # Setup pointer to pre-eval function
    cdef PreEvalFunc pre_eval_func
    
    cdef bint use_drag = True
    if use_drag:
        pre_eval_func = pendulum_preeval_withdrag
    else:
        pre_eval_func = pendulum_preeval_nodrag
    
    # Define time domain
    cdef double t_start = 0.0
    cdef double t_end   = 10.0
    
    # Define initial conditions
    cdef vector[double] y0_vec = vector[double](2)
    y0_vec[0] = 0.01
    y0_vec[1] = 0.0

    # Define our arguments.
    cdef vector[char] args_vec = vector[char](sizeof(double) * 3)
    cdef double* args_dbl_ptr = <double*>args_vec.data()
    args_dbl_ptr[0] = 1.0
    args_dbl_ptr[1] = 1.0
    args_dbl_ptr[2] = 9.81

    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_start,
        t_end,
        y0_vec,
        method = ODEMethod.RK45, # Integration method
        rtol = 1.0e-6,
        atol = 1.0e-8,
        args_vec = args_vec,
        num_extra = 0,
        max_num_steps = 100_000_000,
        max_ram_MB = 2000,
        dense_output = False,
        t_eval_vec = vector[double](), # C++ / Cython require you to provide args in order even if they unused like t_eval_vec here.
        pre_eval_func = pre_eval_func
        )

    # If we want to pass the solution back to python we need to wrap it in a CyRK `WrapCySolverResult` class.
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(move(result))

    return pysafe_result

result_preeval = run()
print("\n\nIntegration success =", result_preeval.success, "\n\tNumber of adaptive time steps required:", result_preeval.size)
print("Integration message:", result_preeval.message)
```
