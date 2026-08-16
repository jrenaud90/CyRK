# Implicit Methods (BDF and LSODA)

A problem is "stiff" when the fastest process in it is much faster than the timescale you actually
care about. The classic symptom is an explicit solver like `RK45` grinding through an enormous
number of very small steps and still reporting success. That is a stability limit, not an accuracy
limit: the step size is being set by the fastest decaying mode rather than by the error tolerance.

Implicit methods are not subject to that limit, so they can take steps sized by accuracy alone. The
price is that every step has to solve a non-linear algebraic system, which means building a
Jacobian matrix and factorizing it.

CyRK provides two implicit methods, both of which follow the implementations in
[SciPy's `solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html):

* **"BDF"** - Implicit multi-step method built on the backward differentiation formulas, with the
  order varying automatically between 1 and 5. It uses a quasi-constant step size and the modified
  (NDF) formulas for extra accuracy.
* **"LSODA"** - A wrapper around ODEPACK's LSODA, which monitors the problem as it integrates and
  switches between the non-stiff Adams formulas and the stiff BDF formulas on its own.

They are used exactly like the explicit methods:

```python
import numpy as np
from CyRK import pysolve_ivp
from numba import njit

@njit
def stiff_diffeq(dy, t, y):
    # The first component relaxes onto cos(t) about 10,000 times faster than the second one moves.
    dy[0] = -1.0e4 * (y[0] - np.cos(t)) - np.sin(t)
    dy[1] = y[0] - y[1]

y0 = np.asarray((1.0, 0.0), dtype=np.float64)

for method in ("RK45", "BDF", "LSODA"):
    result = pysolve_ivp(stiff_diffeq, (0.0, 10.0), y0, method=method,
                         rtol=1.0e-8, atol=1.0e-10, pass_dy_as_arg=True)
    print(f"{method}: {result.steps_taken} steps")

# RK45: 34368 steps
# BDF: 233 steps
# LSODA: 389 steps
```

Everything else in CyRK works with them unchanged: dense output, `t_eval`, events, extra output,
solution reuse, backward integration, and per-variable tolerance arrays.

## Which one should I use?

* If you do not know whether your problem is stiff, use **LSODA**. It starts with the non-stiff
  Adams formulas and only pays for a Jacobian once it detects that it needs one, so it is close to
  an explicit method on non-stiff problems and close to BDF on stiff ones.
* If you know the problem is stiff for its whole domain, **BDF** avoids LSODA's stiffness
  monitoring overhead and tends to be a little more predictable.
* If the problem is not stiff, the explicit methods are still the right answer. `DOP853` in
  particular reaches tight tolerances with far fewer steps than any implicit method.

## The Jacobian

Both methods need the Jacobian matrix of the differential equation, $J_{ij} = \partial \dot{y}_i /
\partial y_j$. By default CyRK estimates it with forward differences, adapting the perturbation
applied to each column so that the difference stays well clear of its own round-off error (this
follows SciPy's `num_jac`).

A finite-difference Jacobian costs one extra differential equation call per dependent variable, so
providing an analytic one is worthwhile for anything but small systems. At the C++ and Cython level
a Jacobian can be handed to `cysolve_ivp` through the `jac_ptr` argument:

```cython
cdef void my_jacobian(double* jac_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil:
    # Column-major (LAPACK) layout: entry (i, j) goes to jac_ptr[i + j * num_y].
    jac_ptr[0] = -1.0e4  # d dy0 / d y0
    jac_ptr[1] = 1.0     # d dy1 / d y0
    jac_ptr[2] = 0.0     # d dy0 / d y1
    jac_ptr[3] = -1.0    # d dy1 / d y1

cdef CySolveOutput result = cysolve_ivp(
    my_diffeq, t_start, t_end, y0_vec, method=ODEMethod.BDF, jac_ptr=my_jacobian)
```

`pysolve_ivp` does not currently accept a Python-level Jacobian; it always uses finite differences.

## Cost and problem size

The implicit methods store the Jacobian and its factorization as dense matrices, so their memory
grows as $N_y^2$ and the cost of each factorization grows as $N_y^3$. CyRK checks the matrix size
against the `max_ram_MB` budget during setup and reports a memory allocation error rather than
attempting a solve that could never finish.

This is a real limit. For a problem with thousands of dependent variables an explicit method is
often the better choice even if the problem is stiff, unless the Jacobian is banded.

## LSODA-only options

LSODA accepts a few options that the other methods do not. Passing them to another method raises an
`AttributeError` rather than silently ignoring them.

| Option | Description |
| --- | --- |
| `min_step` | Smallest step size LSODA is allowed to take. Defaults to 0.0, meaning the step size is bounded only by the error test. |
| `lband`, `uband` | Bandwidth of the Jacobian: entry (i, j) is assumed to be zero unless `i - lband <= j <= i + uband`. Both default to `None`, which treats the Jacobian as dense. |

Declaring a bandwidth is the single most effective thing you can do for a large stiff system whose
variables only couple to their neighbors, such as a discretized PDE. It cuts the number of
differential equation calls needed to build the Jacobian from $N_y$ down to `lband + uband + 1`,
and it replaces the dense factorization with a banded one:

```python
@njit
def diffusion(dy, t, y):
    # Each cell only exchanges with its immediate neighbors, so the Jacobian is tridiagonal.
    num_y = y.size
    for i in range(num_y):
        left = 0.0 if i == 0 else y[i - 1]
        right = 0.0 if i == (num_y - 1) else y[i + 1]
        dy[i] = 50.0 * (left - 2.0 * y[i] + right)

y0 = np.zeros(500, dtype=np.float64)
y0[250] = 100.0

result = pysolve_ivp(diffusion, (0.0, 1.0), y0, method="LSODA",
                     lband=1, uband=1, rtol=1.0e-8, atol=1.0e-10, pass_dy_as_arg=True)
```

For this 500 variable problem both runs take the same 344 steps, but declaring the bandwidth makes
the solve about 20 times faster. The gap widens as the system grows.

At the C++ level these live on the `LSODAConfig` configuration class, which `CySolverResult` builds
automatically when the LSODA method is selected:

```cpp
auto solution_uptr = std::make_unique<CySolverResult>(ODEMethod::LSODA);
LSODAConfig* config_ptr = dynamic_cast<LSODAConfig*>(solution_uptr->config_uptr.get());
config_ptr->num_lower = 1;
config_ptr->num_upper = 1;
config_ptr->min_step_size = 0.0;
config_ptr->max_order_nonstiff = 12;
config_ptr->max_order_stiff = 5;
```

If you supply an analytic Jacobian alongside `lband` and `uband`, it must be written in LAPACK's
band storage: entry (i, j) goes to `jac_ptr[(uband + i - j) + j * (2 * lband + uband + 1)]`.

## Accuracy expectations

For a given `rtol` and `atol` the multi-step methods accumulate somewhat more global error than the
Runge-Kutta methods do. This is inherent to the methods and SciPy's versions behave the same way. If
you are comparing against an analytic solution, expect roughly an order of magnitude more error from
BDF than from RK45 at the same requested tolerance, and tighten the tolerance if that matters.

## Attribution

CyRK's BDF implementation is a C++ port of SciPy's `scipy/integrate/_ivp/bdf.py`. The LSODA method
is built on a modified copy of the C translation of ODEPACK's LSODA that ships with SciPy. See the
`Third-Party Code` section of the [license](License.md) for the full notices, and please cite
ODEPACK if you use the LSODA method.

## References

* Byrne, G. D., and Hindmarsh, A. C., "A Polyalgorithm for the Numerical Solution of Ordinary
  Differential Equations", *ACM Transactions on Mathematical Software*, Vol. 1, No. 1, pp. 71-96,
  1975.
* Shampine, L. F., and Reichelt, M. W., "The MATLAB ODE Suite", *SIAM Journal on Scientific
  Computing*, Vol. 18, No. 1, pp. 1-22, 1997.
* Hairer, E., and Wanner, G., *Solving Ordinary Differential Equations I: Nonstiff Problems*,
  Sec. III.2.
* Hindmarsh, A. C., "ODEPACK, A Systematized Collection of ODE Solvers", *IMACS Transactions on
  Scientific Computation*, Vol. 1, pp. 55-64, 1983.
* Petzold, L., "Automatic selection of methods for solving stiff and nonstiff systems of ordinary
  differential equations", *SIAM Journal on Scientific and Statistical Computing*, Vol. 4, No. 1,
  pp. 136-148, 1983.
