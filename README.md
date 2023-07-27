# CyRK
<div style="text-align: center;">
<a href="https://doi.org/10.5281/zenodo.7093266"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7093266.svg" alt="DOI"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8|3.9|3.10|3.11-blue" alt="Python Version 3.8-3.11" /></a>
<a href="https://codecov.io/gh/jrenaud90/CyRK" ><img src="https://codecov.io/gh/jrenaud90/CyRK/branch/main/graph/badge.svg?token=MK2PqcNGET" alt="Code Coverage"/></a>
<br />
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_win.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_win.yml/badge.svg?branch=main" alt="Windows Tests" /></a>
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_mac.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_mac.yml/badge.svg?branch=main" alt="MacOS Tests" /></a>
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_ubun.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_ubun.yml/badge.svg?branch=main" alt="Ubuntu Tests" /></a>
</div>

---

<a href="https://github.com/jrenaud90/CyRK/releases"><img src="https://img.shields.io/badge/CyRK-0.6.0 Alpha-orange" alt="CyRK Version 0.6.0 Alpha" /></a>


**Runge-Kutta ODE Integrator Implemented in Cython and Numba**

CyRK provides fast integration tools to solve systems of ODEs using an adaptive time stepping scheme. CyRK can, usually, accept differential equation functions 
that are written in pure Python, njited numba, or cython-based cdef classes. These kinds of functions are generally easier to implement than pure c functions. Using CyRK can speed up development time while not making a huge sacrifice when it comes to performance. 

The purpose of this package is to provide some 
functionality of [scipy's solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) with greatly improved performance.

Currently, CyRK's [numba](https://numba.discourse.group/) (njit-safe) implementation is **10-100x faster** than scipy's solve_ivp function.
The [cython](https://cython.org/) `cyrk_ode` function that works with python (or numba) functions is **5-40x faster** than scipy.
The [cython](https://cython.org/) `CySolver` class that works with cython-based cdef classes is **5-400x faster** than scipy.

An additional benefit of the two cython implementations is that they are pre-compiled. This avoids most of the start-up performance hit experienced by just-in-time compilers like numba.

<img style="text-align: center" src="https://github.com/jrenaud90/CyRK/blob/main/Benchmarks/CyRK_SciPy_Compare_v0-6-0a4.png" alt="CyRK Performance" />

## Installation

It is recommended you use an [Anaconda](https://www.anaconda.com/products/distribution) environment. CyRK has been tested on Python 3.8--3.10

To install simply open a terminal and call:

`pip install CyRK`

If not installing from a wheel, CyRK will attempt to install `Cython` and `Numpy` in order to compile the cython code. 
After the files have been compiled, cython will be uninstalled and CyRK's runtime dependencies 
(see the pyproject.toml file for the latest list) will be installed instead.

A new installation of CyRK can be tested quickly by running the following from a python console.
```python
from CyRK import test_cyrk, test_nbrk, test_cysolver
test_cyrk()
# You will hopefully see the message "CyRK's cyrk_ode was tested successfully."
test_nbrk()
# You will hopefully see the message "CyRK's nbrk_ode was tested successfully."
test_cysolver()
# You will hopefully see the message "CyRK's CySolver was tested successfully."
```

### Installation Troubleshooting

*Please [report](https://github.com/jrenaud90/CyRK/issues) installation issues. We will work on a fix and/or add workaround information here.*

- If you see a "Can not load module: CyRK.cy" or similar error then the cython extensions likely did not compile during installation. Try running `pip install CyRK --no-binary="CyRK"` 
to force python to recompile the cython extensions locally (rather than via a prebuilt wheel).
- On MacOS: If you run into problems installing CyRK then reinstall using the verbose flag (`pip install -v .`) to look at the installation log. If you see an error that looks like "clang: error: unsupported option '-fopenmp'" then you may have a problem with your `llvm` or `libomp` libraries. It is recommended that you install CyRK in an anaconda environment with the following packages `conda install numpy scipy cython llvm-openmp`. See more discussion [here](https://github.com/facebookresearch/xformers/issues/157) and the steps taken [here](https://github.com/jrenaud90/CyRK/blob/main/.github/workflows/push_tests_mac.yml).

### Development and Testing Dependencies

If you intend to work on CyRK's code base you will want to install the following dependencies in order to run CyRK's
test suite.

`conda install pytest scipy matplotlib jupyter`

`conda install` can be replaced with `pip install` if you prefer.

## Using CyRK
CyRK's API is similar to SciPy's solve_ivp function. A differential equation can be defined in python such as:

```python
import numpy as np
from numba import njit
# For even more speed up you can use numba's njit to compile the diffeq
@njit
def diffeq_nb(t, y):
    dy = np.empty_like(y)
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy

initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8
```

### Numba-based `nbrk_ode`
The ODE can then be solved using the numba function by calling CyRK's `nbrk_ode`:

```python
from CyRK import nbrk_ode
time_domain, y_results, success, message = \
    nbrk_ode(diffeq_nb, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol)
```

### Cython-based `cyrk_ode`
To call the cython version of the integrator you need to slightly edit the differential equation so that it does not
return the derivative. Instead, the output is passed as an input argument (a np.ndarray) to the function. 

```python
@njit
def diffeq_cy(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
```

Alternatively, you can use CyRK's conversion helper functions to automatically convert between numba/scipy and cyrk
function calls.

```python
from CyRK import nb2cy, cy2nb

@njit
def diffeq_nb(t, y):
    dy = np.empty_like(y)
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy

diffeq_cy = nb2cy(diffeq_nb, use_njit=True)
diffeq_nb2 = cy2nb(diffeq_cy, use_njit=True)
```

You can then call the ODE solver in a similar fashion as the numba version.

```python
from CyRK import cyrk_ode
time_domain, y_results, success, message = \
    cyrk_ode(diffeq_cy, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol)
```

### Cython-based `CySolver`
The cython-based `CySolver` class requires writing a new cython cdef class. This can be done like so, note this is in a .pyx file that must be
cythonized and compiled before it can be used.

```cython
"""ODE.pyx"""
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from CyRK.cy.cysolver cimport CySolver
# Note the `cimport` here^

cdef class CySolverTester(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):
        
        # Unpack dependent variables using the `self.y_new_view` variable.
        cdef double y0, y1
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        # Unpack any additional arguments that do not change with time using the `self.arg_array_view` variable.
        cdef double a, b
        # These must be float64s
        a  = self.arg_array_view[0]
        b  = self.arg_array_view[1]

        # If needed, unpack the time variable using `self.t_new`
        cdef double t
        t = self.t_new

        # This function must set the dydt variable `self.dy_new_view`
        self.dy_new_view[0] = (1. - a * y1) * y0
        self.dy_new_view[1] = (b * y0 - 1.) * y1
```

Once you compile the differential equation it can be imported in a regular python file and used in a similar fashion to the other integrators.

```python
"""run.py"""
from CyRK.cy.cysolvertest import CySolverTester

# Need to make an instance of the integrator.
# The diffeq no longer needs to be passed to the class.
CySolverTesterInst = CySolverTester(time_span, initial_conds, args=(0.01, 0.02), rk_method=1, rtol=rtol, atol=atol)

# To perform the integration make a call to the solve method.
CySolverTesterInst.solve()

# Once complete, you can access the results via...
CySolverTesterInst.success     # True / False
CySolverTesterInst.message     # Note about integration
CySolverTesterInst.solution_t  # Time domain
CySolverTesterInst.solution_y  # y dependent variables
CySolverTesterInst.solution_extra  # Extra output that was captured during integration.
# See Documentation/Extra Output.md for more information on `solution_extra`
```

## Optional Arguments

Both the numba and cython versions of the ODE solver have the following optional inputs:
- `rtol`: Relative Tolerance (default is 1.0e-6).
- `atol`: Absolute Tolerance (default is 1.0e-8).
- `max_step`: Maximum step size (default is +infinity).
- `first_step`: Initial step size (default is 0).
  - If 0, then the solver will try to determine an ideal value.
- `args`: Python tuple of additional arguments passed to the `diffeq`.
- `t_eval`: Both solvers uses an adaptive time stepping protocol based on the recent error at each step. This results in
a final non-uniform time domain of variable size. If the user would like the results at specific time steps then 
they can provide a np.ndarray array at the desired steps to `t_eval`.
The solver will then interpolate the results to fit this array.
- `rk_method`: Runge-Kutta method (default is 1; all of these methods are based off of
[SciPy implementations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)):
  - `0` - "RK23" Explicit Runge-Kutta method of order 3(2).
  - `1` - "RK45" Explicit Runge-Kutta method of order 5(4).
  - `2` - "DOP853" Explicit Runge-Kutta method of order 8.
- `capture_extra` and `interpolate_extra`: CyRK has the capability of capturing additional parameters during integration. Please see `Documentation\Extra Output.md` for more details.

### Additional Arguments for `cyrk_ode` and `CySolver`
- `num_extra` : The number of extra outputs the integrator should expect.
- `expected_size` : Best guess on the expected size of the final time domain (number of points).
    - The integrator must pre-allocate memory to store results from the integration. It will attempt to use arrays sized to `expected_size`. However, if this is too small or too large then performance will be impacted. It is recommended you try out different values based on the problem you are trying to solve.
    - If `expected_size=0` (the default) then the solver will attempt to guess a best size. Currently this is a very basic guess so it is not recommended.
    - It is better to overshoot than undershoot this guess.

## Limitations and Known Issues

- [Issue 1](https://github.com/jrenaud90/CyRK/issues/1): Absolute tolerance can only be passed as a single value
(same for all y's).

## Citing CyRK

It is great to see CyRK used in other software or in scientific studies. We ask that you cite back to CyRK's 
[GitHub](https://github.com/jrenaud90/CyRK) website so interested parties can learn about this package. 

Renaud, Joe P. (2022). CyRK - ODE Integrator Implemented in Cython and Numba. Zenodo. https://doi.org/10.5281/zenodo.7093266

In addition to citing CyRK, please consider citing SciPy and its references for the specific Runge-Kutta model that
was used in your work. CyRK is largely an adaptation of SciPy's functionality.
Find more details [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.

## Contribute to CyRK
_Please look [here](https://github.com/jrenaud90/CyRK/graphs/contributors) for an up-to-date list of contributors to the CyRK package._

CyRK is open-source and is distributed under the Creative Commons Attribution-ShareAlike 4.0 International license. 
You are welcome to fork this repository and make any edits with attribution back to this project (please see the 
`Citing CyRK` section).
- We encourage users to report bugs or feature requests using [GitHub Issues](https://github.com/jrenaud90/CyRK/issues).
- If you would like to contribute but don't know where to start, check out the 
[good first issue](https://github.com/jrenaud90/CyRK/labels/good%20first%20issue) tag on GitHub.
- Users are welcome to submit pull requests and should feel free to create them before the final code is completed so
that feedback and suggestions can be given early on.
