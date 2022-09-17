# CyRK
_Version: 0.0.1 Alpha_

**Runge-Kutta ODE Integrator Implemented in Cython and Numba**

CyRK provides fast ODE integration while still allowing for differential equations written in Python. 

CyRK's numba implementation is 13-25x faster than scipy's solve_ivp function. The cython implementation is about 20x
faster. The cython function is also largely pre-compiled which avoids most of the initial performance hit found
with the numba version.

![CyRK Performance](CyRK_SciPy_Compare_v0-0-1-dev4.png)

## Installation
In order to install `CyRK` you must have the [numpy](https://numpy.org/) and [cython](https://cython.org/) 
packages installed (CyRK will attempt to install these if they are not present). 
It is recommended you pre-install these in an [Anaconda](https://www.anaconda.com/products/distribution) environment.

Once you are ready to install: clone or download the CyRK package, navigate to the directory containing 
CyRK's `setup.py` file, and run the command:

`python -m pip install -e . -v`

This will create a dynamic (editable) link to the CyRK directory so that future updates can be more easily installed.

### Installation Troubleshooting

TBD

### Development and Testing Dependencies

If you intend to work on CyRK's code base you will want to install the following dependencies in order to run CyRK's
test suite.

`conda install pytest scipy matplotlib jupyter`

`conda install` can be replaced with `pip install` if you perfer.

## Using CyRK
CyRK's API is similar to SciPy's solve_ivp function. A differential equation can be defined in python such as:

```python
import numpy as np
from numba import njit
# For even more speed up you can use numba's njit to compile the diffeq
@njit
def diffeq(t, y):
    dy = list()
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
    return dy

initial_conds = np.asarray((20., 20.), dtype=np.complex128)
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8
```

The ODE can then be solved using the numba function by calling CyRK's `nbrk_ode`:

```python
from CyRK import nbrk_ode
time_domain, y_results, success, message = \
    nbrk_ode(diffeq, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol)
```

To call the cython version of the integrator you need to slightly edit the differential equation so that it does not
return the derivative. Instead, the output is passed as an input argument (a np.ndarray) to the function. 

```python
@njit
def diffeq(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
```

You can then call the ODE solver in a similar fashion as the numba version.

```python
from CyRK import cyrk_ode
time_domain, y_results, success, message = \
    cyrk_ode(diffeq, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol)
```

### Optional Inputs

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

### Limitations and Known Issues

- [Issue 1](https://github.com/jrenaud90/CyRK/issues/1): Absolute tolerance can only be passed as a single value
(same for all y's).
- [Issue 3](https://github.com/jrenaud90/CyRK/issues/3): Right now the cython version only allows for complex-valued
y-values.
- [Issue 5](https://github.com/jrenaud90/CyRK/issues/5): The numba solver is worse than the pure python scipy solver at
large timespans (high integration times).

## Citing CyRK

It is great to see CyRK used in other software or in scientific studies. We ask that you cite back to CyRK's 
[GitHub](https://github.com/jrenaud90/CyRK) website so more users learn about this package.

In addition to citing CyRK, please consider citing SciPy and its references for the specific Runge-Kutta model that
was used in your work. CyRK is largely an adaptation of SciPy's functionality.
Find more details [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

## Contribute to CyRK
CyRK is open-source and is distributed under the Creative Commons Attribution-ShareAlike 4.0 International license. 
You are welcome to fork this repository and make any edits with attribution back to this project (please see the 
`Citing CyRK` section).
- We encourage users to report bugs or feature requests using [GitHub Issues](https://github.com/jrenaud90/CyRK/issues).
- If you would like to contribute but don't know where to start, check out the 
[good first issue](https://github.com/jrenaud90/CyRK/labels/good%20first%20issue) tag on GitHub.
- Users are welcome to submit pull requests and should feel free to create them before the final code is completed so
that feedback and suggestions can be given early on.
