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
**Deprecation Warning:** CySolver class based method was a previous version of CyRK's cython solver for cython-only functions. It has been replaced by `from CyRK cimport cysolve_ivp`. The original `CySolver` is no longer supported and has been removed.

There still exists a C++ `CySolver` class that is used in CyRK's backend. We also will often refer to the C++ backend
as a whole colloquially as `CySolver`. 
