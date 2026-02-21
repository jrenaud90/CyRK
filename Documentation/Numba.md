# Numba `nbsolve_ivp`

:::{alert}
**Deprecation Warning!** As of CyRK v0.17.0 the original `CyRK.nbsolve_ivp` is marked for deprecation. In a future
release it will be replaced by the new `CyRK.nbsolve2_ivp`. These two functions have similar purposes but different
calling methods and very different internals. Details about both can be found on this page. As of CyRK v0.17.0,
the current `nbsolve_ivp` will print out a warning message discussing this deprecation. These messages have a negative
impact on performance and can be disabled by passing `warnings=False` to the solver.
:::