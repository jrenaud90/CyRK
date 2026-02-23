# CyRK's Deprecated Methods
The functions discussed in this document were packaged in previous version of CyRK (pre v0.11.0) but are no longer available (as of CyRK v0.11.0).

Documentation is retained for comparison purposes to the new methods. This documentation will be removed in a future version of CyRK.

## Numba-Wrapped `nbsolve_ivp`
:::{attention}
**Deprecation Warning!** As of CyRK v0.17.0 the original `CyRK.nbsolve_ivp` is marked for deprecation. In a future
release it will be replaced by the new `CyRK.nbsolve2_ivp`. These two functions have similar purposes but different
calling methods and very different internals. Details about both can be found on the [Numba page](Numba.md).
As of CyRK v0.17.0, the current `nbsolve_ivp` will print out a warning message discussing this deprecation.
These messages have a negative impact on performance and can be disabled by passing `warnings=False` to the solver.
:::

## Cython-based `cyrk_ode`
**Deprecation Warning:** cyrk_ode was a previous version of CyRK's cython solver that could take in python functions.
It is no longer supported and has been removed as of CyRK version v0.11.0. 
To call the new cython-based solver, `pysolve_ivp` you will need to edit your differential equation and other
input parameters. Please review the [Getting Started](Readme.md) page.


## Cython-based `CySolver`
**Deprecation Warning:** CySolver class based method was a previous version of CyRK's cython solver for cython-only
functions. It has been replaced by `from CyRK cimport cysolve_ivp`. The original `CySolver` is no longer supported
and has been removed. Please review the [Getting Started](Readme.md) page and [C++ API page](C++_API.md) to learn
how to use this new solver.

There still exists a C++ `CySolver` class that is used in CyRK's backend. We also will often refer to the C++ backend
as a whole colloquially as `CySolver`. 
