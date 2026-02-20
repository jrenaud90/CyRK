# Parallelizing CyRK

_Note the discussion below only pertains to CyRK's `cysolve_ivp`, `pysolve_ivp`, and related methods. `nbsolve_ivp` may support parallelization but it has not been tested thoroughly and is not officially supported._

The inner workings of CySolver are not parallelized on purpose: generally the performance gains of parallelizing the
integration steps are far out weighed by the complexity, errors, and most importantly, overhead of distributed work.
However, the functions that interact with that backend (`pysolve_ivp`, `cysolve_ivp`, and their derivatives) can
be use in parallelized loops. This can greatly speed up programs that have many, slow integrations.

## Parallel `pysolve_ivp`
`pysolve_ivp` function can be parallelized using Python's [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) package.
Note that it can not utilize multithreading because `pysolve_ivp` requires a reference to the python differential
equation. This would be shared across threads leading to inadvertent serialization if not just crashing.
Examples on how this is done can be found in the Getting Started notebook.

## Parallel `cysolve_ivp`
`cysolve_ivp` function can be parallelized using Cython's [prange](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html).
Examples on how this is done can be found in the Advanced CySolver Examples notebook or in `CyRK.cy.prange_test`.
