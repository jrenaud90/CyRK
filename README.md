# CyRK
<div style="text-align: center;">
<a href="https://doi.org/10.5281/zenodo.7093266"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7093266.svg" alt="DOI"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8|3.9|3.10|3.11|3.12-blue" alt="Python Version 3.8-3.12" /></a>
<!-- <a href="https://codecov.io/gh/jrenaud90/CyRK" ><img src="https://codecov.io/gh/jrenaud90/CyRK/branch/main/graph/badge.svg?token=MK2PqcNGET" alt="Code Coverage"/></a> -->
<br />
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_win.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_win.yml/badge.svg?branch=main" alt="Windows Tests" /></a>
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_mac.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_mac.yml/badge.svg?branch=main" alt="MacOS Tests" /></a>
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_ubun.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_ubun.yml/badge.svg?branch=main" alt="Ubuntu Tests" /></a>

<br />
<a href="https://pypi.org/project/CyRK/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/CyRK?label=PyPI%20Downloads" /></a>
</div>

---

<a href="https://github.com/jrenaud90/CyRK/releases"><img src="https://img.shields.io/badge/CyRK-0.12.2 Alpha-orange" alt="CyRK Version 0.12.2 Alpha" /></a>


**Runge-Kutta ODE Integrator Implemented in Cython and Numba**

CyRK provides fast integration tools to solve systems of ODEs using an adaptive time stepping scheme. CyRK can accept differential equations that are written in pure Python, njited numba, or cython-based cdef functions. These kinds of functions are generally easier to implement than pure c functions and can be used in existing Python software. Using CyRK can speed up development time while avoiding the slow performance that comes with using pure Python-based solvers like SciPy's `solve_ivp`.

The purpose of this package is to provide some 
functionality of [scipy's solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) with greatly improved performance.

Currently, CyRK's [numba-based](https://numba.discourse.group/) (njit-safe) implementation is **10-140x faster** than scipy's solve_ivp function.
The [cython-based](https://cython.org/) `pysolve_ivp` function that works with python (or njit'd) functions is **20-50x faster** than scipy.
The [cython-based](https://cython.org/) `cysolver_ivp` function that works with cython-based cdef functions is **50-700+x faster** than scipy.

An additional benefit of the two cython implementations is that they are pre-compiled. This avoids most of the start-up performance hit experienced by just-in-time compilers like numba.


<img style="text-align: center" src="https://github.com/jrenaud90/CyRK/blob/main/Benchmarks/CyRK_SciPy_Compare_predprey_v0-12-0.png" alt="CyRK Performance Graphic" />

## Installation

*CyRK has been tested on Python 3.8--3.12; Windows, Ubuntu, and MacOS.*

Install via pip:

`pip install CyRK`

conda:

`conda install -c conda-forge CyRK`

mamba:

`mamba install cyrk`

If not installing from a wheel, CyRK will attempt to install `Cython` and `Numpy` in order to compile the source code. A "C++ 20" compatible compiler is required.
Compiling CyRK has been tested on the latest versions of Windows, Ubuntu, and MacOS. Your milage may vary if you are using a older or different operating system.
If on MacOS you will likely need a non-default compiler in order to compile the required openMP package. See the "Installation Troubleshooting" below. 
After everything has been compiled, cython will be uninstalled and CyRK's runtime dependencies (see the pyproject.toml file for the latest list) will be installed instead.

A new installation of CyRK can be tested quickly by running the following from a python console.
```python
from CyRK import test_pysolver, test_cysolver, test_nbrk
test_pysolver()
# Should see "CyRK's PySolver was tested successfully."
test_cysolver()
# Should see "CyRK's CySolver was tested successfully."
test_nbrk()
# Should see "CyRK's nbrk_ode was tested successfully."
```

### Installation Troubleshooting

*Please [report](https://github.com/jrenaud90/CyRK/issues) installation issues. We will work on a fix and/or add workaround information here.*

- If you see a "Can not load module: CyRK.cy" or similar error then the cython extensions likely did not compile during installation. Try running `pip install CyRK --no-binary="CyRK"` 
to force python to recompile the cython extensions locally (rather than via a prebuilt wheel).

- On MacOS: If you run into problems installing CyRK then reinstall using the verbose flag (`pip install -v .`) to look at the installation log. If you see an error that looks like "clang: error: unsupported option '-fopenmp'" then you are likely using the default compiler or other compiler that does not support OpenMP. Read more about this issue [here](https://github.com/facebookresearch/xformers/issues/157) and the steps taken [here](https://github.com/jrenaud90/CyRK/blob/main/.github/workflows/push_tests_mac.yml). A fix for this issue is to use `llvm`'s clang compiler. This can be done by doing the following in your terminal before installing CyRK.
```
brew install llvm
brew install libomp

# If on ARM64 (Apple Silicon) then do:
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
# Otherwise change these directories to:
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export LDFLAGS="-L/usr/local/opt/libomp/lib"
export CPPFLAGS="-I/usr/local/opt/libomp/include"
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++

pip install CyRK --no-binary="CyRK"
```

- CyRK has a number of runtime status codes which can be used to help determine what failed during integration. Learn more about these codes [https://github.com/jrenaud90/CyRK/blob/main/Documentation/Status%20and%20Error%20Codes.md](here).

### Development and Testing Dependencies

If you intend to work on CyRK's code base you will want to install the following dependencies in order to run CyRK's test suite and experimental notebooks.

`conda install pytest scipy matplotlib jupyter`

`conda install` can be replaced with `pip install` if you prefer.

## Using CyRK

**The following code can be found in a Jupyter Notebook called "Getting Started.ipynb" in the "Demos" folder.**

*Note: some older CyRK functions like `cyrk_ode` and `CySolver` class-based method have been deprecated and removed. Read more in "Documentation/Deprecations.md".*
CyRK's API is similar to SciPy's solve_ivp function. A differential equation can be defined in python such as:

```python
# For even more speed up you can use numba's njit to compile the diffeq
from numba import njit
@njit
def diffeq_nb(t, y):
    dy = np.empty_like(y)
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy
```

### Numba-based `nbsolve_ivp`

_Future Development Note: The numba-based solver is currently in a feature-locked state and will not receive new features (as of CyRK v0.9.0). The reason for this is because it uses a different backend than the rest of CyRK and is not as flexible or easy to expand without significant code duplication. Please see GitHub Issue: TBD to see the status of this new numba-based solver or share your interest in continued development._

The system of ODEs can then be solved using CyRK's numba solver by,

```python
import numpy as np
from CyRK import nbsolve_ivp

initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8

result = \
    nbsolve_ivp(diffeq_nb, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol)

print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(result.t, result.y[0], c='r')
ax.plot(result.t, result.y[1], c='b')
```

#### `nbsolve_ivp` Arguments
```python
nbsolve_ivp(
    diffeq: callable,                  # Differential equation defined as a numba.njit'd python function
    t_span: Tuple[float, float],       # Python tuple of floats defining the start and stop points for integration
    y0: np.ndarray,                    # Numpy array defining initial y0 conditions.
    args: tuple = tuple(),             # Python Tuple of additional args passed to the differential equation. These can be any njit-safe object.
    rtol: float = 1.e-3,               # Relative tolerance used to control integration error.
    atol: float = 1.e-6,               # Absolute tolerance (near 0) used to control integration error.
    rtols: np.ndarray = EMPTY_ARR,     # Overrides rtol if provided. Array of floats of rtols if you'd like a different rtol for each y.
    atols: np.ndarray = EMPTY_ARR,     # Overrides atol if provided. Array of floats of atols if you'd like a different atol for each y.
    max_step: float = np.inf,          # Maximum allowed step size.
    first_step: float = None,          # Initial step size. If set to 0.0 then CyRK will guess a good step size.
    rk_method: int = 1,                # Integration method. Current options: 0 == RK23, 1 == RK45, 2 == DOP853
    t_eval: np.ndarray = EMPTY_ARR,    # `nbsolve_ivp` uses an adaptive time stepping protocol based on the recent error at each step. This results in a final non-uniform time domain of variable size. If the user would like the results at specific time steps then they can provide a np.ndarray array at the desired steps via `t_eval`. The solver will then interpolate the results to fit this 
    capture_extra: bool = False,       # Set to True if the diffeq is designed to provide extra outputs.
    interpolate_extra: bool = False,   # See "Documentation/Extra Output.md" for details.
    max_num_steps: int = 0             # Maximum number of steps allowed. If exceeded then integration will fail. 0 (the default) turns this off.
    )
```

### Python wrapped `pysolve_ivp`

CyRK's main integration functions utilize a C++ backend system which is then wrapped and accessible to Python via Cython. The easiest way to interface with this system is through CyRK's `pysolve_ivp` function. It follows a very similar format to both `nbsolve_ivp` and Scipy's `solve_ivp`. First you must build a function in Python. This could look the same as the function described above for `nbsolve_ivp` (see `diffeq_nb`). However, there are a few advantages that pysolve_ivp provides over nbsolve_ivp:

  1. It accepts both functions that use numba's njit wrapper (as `diffeq_nb` did above) or pure Python functions (`nbsolve_ivp` only accepts njit'd functions).
  2. You can provide the resultant dy/dt as an argument which can provide a significant performance boost.

Utilizing point 2, we can re-write the differential equation function as,

```python
# Note if using this format, `dy` must be the first argument. Additionally, a special flag must be set to True when calling pysolve_ivp, see below.
def cy_diffeq(dy, t, y):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
```

Since this function is not using any special functions we could easily wrap it with njit for additional performance boost: `cy_diffeq = njit(cy_diffeq)`.

Once you have built your function the procedure to solve it is:

```python
import numpy as np
from CyRK import pysolve_ivp

initial_conds = np.asarray((20., 20.), dtype=np.complex128, order='C')
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8

result = \
    pysolve_ivp(cy_diffeq, time_span, initial_conds, method="RK45", rtol=rtol, atol=atol,
                # Note if you did build a differential equation that has `dy` as the first argument then you must pass the following flag as `True`.
                # You could easily pass the `diffeq_nb` example which returns dy. You would just set this flag to False (and experience a hit to your performance).
                pass_dy_as_arg=True)

print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(result.t, result.y[0], c='r')
ax.plot(result.t, result.y[1], c='b')
```

#### `pysolve_ivp` Arguments

```python
def pysolve_ivp(
        object py_diffeq,            # Differential equation defined as a python function
        tuple time_span,             # Python tuple of floats defining the start and stop points for integration
        double[::1] y0,              # Numpy array defining initial y0 conditions.
        str method = 'RK45',         # Integration method. Current options are: RK23, RK45, DOP853
        double[::1] t_eval = None,   # Array of time steps at which to save data. If not provided then all adaptive time steps will be saved. There is a slight performance hit using this feature.
        bint dense_output = False,   # If True, then dense interpolators will be saved to the solution. This allows a user to call solution as if a function (in time).
        tuple args = None,           # Python Tuple of additional args passed to the differential equation. These can be any python object.
        size_t expected_size = 0,    # Expected size of the solution. There is a slight performance improvement if selecting the the exact or slightly more time steps than the adaptive stepper will require (if you happen to know this ahead of time).
        size_t num_extra = 0,  # Number of extra outputs you want to capture during integration. There is a performance hit if this is used in conjunction with t_eval or dense_output.
        double first_step = 0.0,     # Initial step size. If set to 0.0 then CyRK will guess a good step size.
        double max_step = INF,       # Maximum allowed step size.
        rtol = 1.0e-3,               # Relative tolerance used to control integration error. This can be provided as a numpy array if you'd like a different rtol for each y.
        atol = 1.0e-6,               # Absolute tolerance (near 0) used to control integration error. This can be provided as a numpy array if you'd like a different atol for each y.
        size_t max_num_steps = 0,    # Maximum number of steps allowed. If exceeded then integration will fail. 0 (the default) turns this off.
        size_t max_ram_MB = 2000,    # Maximum amount of system memory the integrator is allowed to use. If this is exceeded then integration will fail.
        bint pass_dy_as_arg = False  # Flag if differential equation returns dy (False) or is passed dy as the _first_ argument (True).
        ):
```

### Pure Cython `cysolve_ivp`

A final method is provided to users in the form of `cysolve_ivp`. This function can only be accessed and used by code written in Cython. Details about how to setup and use Cython can be found on the project's [website](https://cython.readthedocs.io/en/stable/index.html). The below code examples assume you are running the code in a [Jupyter Notebook](https://jupyter.org/).

`cysolve_ivp` has a slightly different interface than `nbsolve_ivp` and `pysolve_ivp` as it only accepts C types. For that reason, python functions will not work with `cysolve_ivp`. While developing in Cython is more challenging than Python, there is a huge performance advantage (`cysolve_ivp` is roughly 5x faster than `pysolve_ivp` and 700x faster than scipy's `solve_ivp`). Below is a demonstration of how it can be used.

First a pure Cython file (written as a Jupyter notebook).
```cython
%%cython --force 
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
np.import_array()

# Note the "distutils" and "cython" headers above are functional. They tell cython how to compile the code. In this case we want to use C++ and to turn off several safety checks (which improve performance).

# The cython diffeq is much less flexible than the others described above. It must follow this format, including the type information. 
# Currently, CyRK only allows additional arguments to be passed in as a double array pointer (they all must be of type double). Mixed type args will be explored in the future if there is demand for it (make a GitHub issue if you'd like to see this feature!).
# The "noexcept nogil" tells cython that the Python Global Interpretor Lock is not required, and that no exceptions should be raised by the code within this function (both improve performance).
# If you do need the gil for your differential equation then you must use the `cysolve_ivp_gil` function instead of `cysolve_ivp`

# Import the required functions from CyRK
from CyRK cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput, PreEvalFunc

# Note that currently you must provide the "char* args, PreEvalFunc pre_eval_func" as inputs even if they are unused.
# See "Advanced CySolver.md" in the documentation for information about these parameters.
cdef void cython_diffeq(double* dy, double t, double* y, char* args, PreEvalFunc pre_eval_func) noexcept nogil:
    
    # Unpack args
    # CySolver assumes an arbitrary data type for additional arguments. So we must cast them to the array of 
    # doubles that we want to use for this equation
    cdef double* args_as_dbls = <double*>args
    cdef double a = args_as_dbls[0]
    cdef double b = args_as_dbls[1]
    
    # Build Coeffs
    cdef double coeff_1 = (1. - a * y[1])
    cdef double coeff_2 = (b * y[0] - 1.)
    
    # Store results
    dy[0] = coeff_1 * y[0]
    dy[1] = coeff_2 * y[1]
    # We can also capture additional output with cysolve_ivp.
    dy[2] = coeff_1
    dy[3] = coeff_2

# Import the required functions from CyRK
from CyRK cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput

# Let's get the integration number for the RK45 method
from CyRK cimport RK45_METHOD_INT

# Now let's import cysolve_ivp and build a function that runs it. We will not make this function `cdef` like the diffeq was. That way we can run it from python (this is not a requirement. If you want you can do everything within Cython).
# Since this function is not `cdef` we can use Python types for its input. We just need to clean them up and convert them to pure C before we call cysolve_ivp.
def run_cysolver(tuple t_span, double[::1] y0):
    
    # Cast our diffeq to the accepted format
    cdef DiffeqFuncType diffeq = cython_diffeq
    
    # Convert the python user input to pure C types
    cdef double* y0_ptr       = &y0[0]
    cdef size_t num_y         = len(y0)
    cdef double[2] t_span_arr = [t_span[0], t_span[1]]
    cdef double* t_span_ptr   = &t_span_arr[0]

    # Assume constant args
    cdef double[2] args   = [0.01, 0.02]
    cdef double* args_dbl_ptr = &args[0]
    # Need to cast the arg double pointer to void
    cdef char* args_ptr = <char*>args_dbl_ptr

    # CySolver makes a copy of the arg structure so that it can retain the values for later computation.
    # In order to make this copy, it needs to know the size of the structure.
    cdef size_t size_of_args = 2 * sizeof(double)

    # Run the integrator!
    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_span_ptr,
        y0_ptr,
        num_y,
        method = RK45_METHOD_INT, # Integration method
        rtol = 1.0e-7,
        atol = 1.0e-8,
        args_ptr = args_ptr,
        size_of_args = size_of_args,
        num_extra = 2
    )

    # The CySolveOutput is not accesible via Python. We need to wrap it first
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(result)

    return pysafe_result
```

Now we can make a python script that calls our new cythonized wrapper function. Everything below is in pure Python.

```python
# Assume we are working in a Jupyter notebook so we don't need to import `run_cysolver` if it was defined in an earlier cell.
# from my_cython_code import run_cysolver

import numpy as np
initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
time_span = (0., 50.)

result = run_cysolver(time_span, initial_conds)

print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(result.t, result.y[0], c='r')
ax.plot(result.t, result.y[1], c='b')

# Can also plot the extra output. They are small for this example so scaling them up by 100
ax.plot(result.t, 100*result.y[2], c='green', ls=':')
ax.plot(result.t, 100*result.y[3], c='purple', ls=':')
```

There is a lot more you can do to interface with CyRK's C++ backend and fully optimize the integrators to your needs. These details will be documented in "Documentation/Advanced CySolver.md".

#### No Return `cysolve_ivp_noreturn`
The example above shows using `cysolve_ivp` functions that return an output (which is a c++ shared pointer): `cdef CySolveOutput result = cysolve_ivp(...)`. CyRK also provides a function that takes the output as an input if you prefer to manage your own memory:
```cython

from libcpp.memory cimport make_shared, shared_ptr

from CyRK cimport CySolverResult, cysolve_ivp_noreturn

# Make our own storage
cdef shared_ptr[CySolverResult] solution_sptr =
    make_shared[CySolverResult](
        num_y,
        num_extra,
        expected_size,
        t_end,
        direction_flag,
        dense_output,
        t_eval_provided);

# Pass it to the noreturn version of the solver for it to update.
cysolve_ivp_noreturn(solution_sptr, <other inputs>)

```

#### `cysolve_ivp` and `cysolve_ivp_gil` Arguments

```cython
cdef shared_ptr[CySolverResult] cysolve_ivp(
    DiffeqFuncType diffeq_ptr,        # Differential equation defined as a cython function
    double* t_span_ptr,               # Pointer to array (size 2) of floats defining the start and stop points for integration
    double* y0_ptr,                   # Pointer to array defining initial y0 conditions.
    size_t num_y,                     # Size of y0_ptr array.
    int method = 1,                   # Integration method. Current options: 0 == RK23, 1 == RK45, 2 == DOP853
    double rtol = 1.0e-3,             # Relative tolerance used to control integration error.
    double atol = 1.0e-6,             # Absolute tolerance (near 0) used to control integration error.
    char* args_ptr = NULL,            # Pointer to array of additional arguments passed to the diffeq. See "Advanced CySolver.md" for more details.
    size_t size_of_args = 0,          # Size of the structure that `args_ptr` is pointing to. 
    size_t num_extra = 0,             # Number of extra outputs you want to capture during integration. There is a performance hit if this is used in conjunction with t_eval or dense_output.
    size_t max_num_steps = 0,         # Maximum number of steps allowed. If exceeded then integration will fail. 0 (the default) turns this off.
    size_t max_ram_MB = 2000,         # Maximum amount of system memory the integrator is allowed to use. If this is exceeded then integration will fail.
    bint dense_output = False,        # If True, then dense interpolators will be saved to the solution. This allows a user to call solution as if a function (in time).
    double* t_eval = NULL,            # Pointer to an array of time steps at which to save data. If not provided then all adaptive time steps will be saved. There is a slight performance hit using this feature.
    size_t len_t_eval = 0,            # Size of t_eval.
    PreEvalFunc pre_eval_func = NULL  # Optional additional function that is called within `diffeq_ptr` using current `t` and `y`. See "Advanced CySolver.md" for more details.
    double* rtols_ptr = NULL,         # Overrides rtol if provided. Pointer to array of floats of rtols if you'd like a different rtol for each y.
    double* atols_ptr = NULL,         # Overrides atol if provided. Pointer to array of floats of atols if you'd like a different atol for each y.
    double max_step = MAX_STEP,       # Maximum allowed step size.
    double first_step = 0.0           # Initial step size. If set to 0.0 then CyRK will guess a good step size.
    size_t expected_size = 0,         # Expected size of the solution. There is a slight performance improvement if selecting the the exact or slightly more time steps than the adaptive stepper will require (if you happen to know this ahead of time).
    )
```

## Limitations and Known Issues

- [Issue 30](https://github.com/jrenaud90/CyRK/issues/30): CyRK's cysolve_ivp and pysolve_ivp does not allow for complex-valued dependent variables. 

## Citing CyRK

It is great to see CyRK used in other software or in scientific studies. We ask that you cite back to CyRK's [GitHub](https://github.com/jrenaud90/CyRK) website so interested parties can learn about this package. It would also be great to hear about the work being done with CyRK, so get in touch!

Renaud, Joe P. (2022). CyRK - ODE Integrator Implemented in Cython and Numba. Zenodo. https://doi.org/10.5281/zenodo.7093266

In addition to citing CyRK, please consider citing SciPy and its references for the specific Runge-Kutta model that was used in your work. CyRK is largely an adaptation of SciPy's functionality. Find more details [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.

## Contribute to CyRK
_Please look [here](https://github.com/jrenaud90/CyRK/graphs/contributors) for an up-to-date list of contributors to the CyRK package._

CyRK is open-source and is distributed under the Creative Commons Attribution-ShareAlike 4.0 International license. You are welcome to fork this repository and make any edits with attribution back to this project (please see the `Citing CyRK` section).
- We encourage users to report bugs or feature requests using [GitHub Issues](https://github.com/jrenaud90/CyRK/issues).
- If you would like to contribute but don't know where to start, check out the [good first issue](https://github.com/jrenaud90/CyRK/labels/good%20first%20issue) tag on GitHub.
- Users are welcome to submit pull requests and should feel free to create them before the final code is completed so that feedback and suggestions can be given early on.
