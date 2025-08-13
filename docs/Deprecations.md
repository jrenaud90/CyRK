# CyRK's Deprecated Methods
The functions discussed in this document were packaged in previous version of CyRK (pre v0.11.0) but are no longer available (as of CyRK v0.11.0).

Documentation is retained for comparison purposes to the new methods. This documentation will be removed in a future version of CyRK.

## Cython-based `cyrk_ode`
**Deprecation Warning:** cyrk_ode was a previous version of CyRK's cython solver that could take in python functions. It is no longer supported and will be removed in a future version of CyRK.
To call the cython version of the integrator you need to slightly edit the differential equation so that it does not
return the derivative. Instead, the output is passed as an input argument (a `np.ndarray`) to the function. 

```python
@njit
def diffeq_cy(t, y, dy):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]
```

Alternatively, you can use CyRK's conversion helper functions to automatically convert between numba/scipy and cyrk function calls.

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

## Cython-based `CySolver`
**Deprecation Warning:** CySolverc class based method was a previous version of CyRK's cython solver for cython-only functions. It has been replaced by `from CyRK cimport cysolve_ivp`. `CySolver` is no longer supported and will be removed in a future version of CyRK.
The cython-based `CySolver` class requires writing a new cython cdef class. This is done in a new cython .pyx file which must then be cythonized and compiled before it can be used.

```cython
"""ODE.pyx"""
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from CyRK.cy.cysolver cimport CySolver
# Note the `cimport` here^

cdef class MyCyRKDiffeq(CySolver):

    cdef void diffeq(self) noexcept nogil:
        
        # Unpack dependent variables using the `self.y_ptr` variable.
        # In this example we have a system of two dependent variables, but any number can be used.
        cdef double y0, y1
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]

        # Unpack any additional arguments that do not change with time using the `self.args_ptr` variable.
        cdef double a, b
        # These must be float64s
        a  = self.args_ptr[0]
        b  = self.args_ptr[1]

        # If needed, unpack the time variable using `self.t_now`
        cdef double t
        t = self.t_now

        # This then updates dydt by setting the values of `self.dy_ptr`
        self.dy_ptr[0] = (1. - a * y1) * y0
        self.dy_ptr[1] = (b * y0 - 1.) * y1
```

Once you compile the differential equation it can be imported in a regular python file and used in a similar fashion to the other integrators.

```python
"""run.py"""
from ODE import MyCyRKDiffeq

# It is important that any arrays passed to the CySolver are C-contiguous (set with numpy with "order=C")
# Also, currently, CySolver only works with floats/doubles. Not complex.
initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')

# Need to make an instance of the integrator.
# The diffeq no longer needs to be passed to the class.
MyCyRKDiffeqInst = MyCyRKDiffeq(time_span, initial_conds, args=(0.01, 0.02), rk_method=1, rtol=rtol, atol=atol, auto_solve=True)

# To perform the integration make a call to the solve method.
# Only required if the `auto_solve` flag is set to False (defaults to True)
# MyCyRKDiffeqInst.solve()

# Once complete, you can access the results via...
MyCyRKDiffeqInst.success     # True / False
MyCyRKDiffeqInst.message     # Note about integration
MyCyRKDiffeqInst.t           # Time domain
MyCyRKDiffeqInst.y           # y dependent variables
MyCyRKDiffeqInst.extra       # Extra output that was captured during integration.
# See docs/Extra Output.md for more information on `extra`
```