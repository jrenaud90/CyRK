# Integration Methods

CyRK provides six explicit Runge-Kutta methods for non-stiff problems and three implicit methods for stiff problems. Every method is selected with the `method` argument of `pysolve_ivp` and `nbsolve2_ivp` (a case-insensitive string such as `"Vern7"`) or of `cysolve_ivp` (the C++ enum `ODEMethod`, such as `ODEMethod.VERN7`), and every method supports dense output, `t_eval`, events, extra output, solution reuse, backward integration, and per-variable tolerances. The explicit methods use the same step size controller and error norm as SciPy's `solve_ivp`.

## Methods

| Method | Type | Order (error estimate) | Diffeq calls per step | Interpolant order (extra calls) | Memory per $y$ \[bytes\] |
| --- | --- | --- | --- | --- | --- |
| `RK23` | Explicit RK, Bogacki and Shampine | 3 (2) | 3 | 3 (0) | 112 |
| `RK45` | Explicit RK, Dormand and Prince | 5 (4) | 6 | 4 (0) | 136 |
| `DOP853` | Explicit RK, Dormand and Prince | 8 (5 and 3) | 12 | 7 (3) | 224 |
| `Tsit5` | Explicit RK, Tsitouras | 5 (4) | 6 | 4 (0) | 136 |
| `Vern7` | Explicit RK, Verner | 7 (6) | 10 | 7 (5) | 208 |
| `Vern8` | Explicit RK, Verner | 8 (7) | 13 | 8 (7) | 248 |
| `BDF` | Implicit multistep (NDF) | 1 to 5, variable | 1 or more per Newton iteration | current order (0) | grows with $N_y$ |
| `Radau` | Implicit RK, Radau IIA | 5 (3) | 3 or more per Newton iteration | 3 (0) | grows with $N_y$ |
| `LSODA` | Adams or BDF, switching | 1 to 12 (Adams), 1 to 5 (BDF) | 1 or more per corrector iteration | current order (0) | grows with $N_y$ |

The explicit methods reuse the derivative at the end of each step as the first stage of the next. "Extra calls" are made on each step that builds an interpolant, which is every step when `dense_output=True` or when there are events, and only the steps that contain a `t_eval` point otherwise. The implicit methods also call the diffeq $N_y$ times each time they estimate the Jacobian by finite differences. The memory column is the solver's footprint per dependent variable. The implicit methods store an $N_y$ by $N_y$ Jacobian and its factorization, so their footprint per variable grows with $N_y$ (see the [C++ API](C++_API.md) page for the full formulas).

### Explicit Runge-Kutta Methods

* `RK23` and `RK45` follow SciPy's methods of the same names. `RK45` is the default.
* `DOP853` is Hairer's DOP853 as implemented in SciPy. Its error estimate combines fifth- and third-order formulas, and its seventh-order interpolant needs three extra stages.
* `Tsit5` is the 5(4) pair of Tsitouras (2011). It costs the same per step as `RK45`, but its error coefficients are smaller, and its fourth-order interpolant uses only the seven stages that the step already computed.
* `Vern7` and `Vern8` are Verner's "most efficient" 7(6) and 8(7) pairs (Verner 2010). Each has one stage that is used only by the error estimate (stage 10 of `Vern7`, stage 13 of `Vern8`). Their interpolants match the order of the method and need five (`Vern7`) or seven (`Vern8`) extra stages, which are only evaluated on steps that need an interpolant. `Vern7` uses the interpolant coefficients as Verner corrected them in 2024.

### Implicit Methods

`BDF`, `Radau`, and `LSODA` solve a non-linear system at each step and so are not limited by stability on stiff problems. The [Implicit Methods](Implicit_Methods.md) page describes them, their options, and how to choose between them.

## Choosing a Method

The right method depends mostly on how accurate the answer needs to be, how expensive the differential equation is to evaluate, and whether the problem is stiff. The suggestions below are drawn from the measurements in the next section and from the references. They are not universal, so a quick comparison on your own problem is always a good idea.

* **Stiff problems**: if an explicit method takes a very large number of steps even at loose tolerances, the problem is likely stiff and its steps are limited by stability rather than accuracy. Use `BDF` or `Radau` (see [Implicit Methods](Implicit_Methods.md)).
* **Unknown or changing stiffness**: `LSODA` starts with the non-stiff Adams formulas and switches to BDF when it detects stiffness, so it is a reasonable choice when you do not know which kind of problem you have, or when it changes along the integration.
* **Low accuracy (relative errors of about 1e-3 to 1e-5)**: the high-order methods are often the cheapest even here (`Vern8` needed the fewest diffeq calls at an error of 1e-4 on four of the five problems below, and `DOP853` on the fifth). The low-order methods win mainly when the solution is not smooth (a diffeq with kinks or discontinuities lowers the effective order of every method) or when the steps are limited by stability. Between the two fifth-order methods, `Tsit5` typically needs fewer calls than `RK45`: on the problems below it needed 7% to 49% fewer at errors of 1e-6 and smaller, and 9% to 23% fewer at 1e-4 except on the Arenstorf orbit, where the two were equal. It is not always cheaper: on a Kepler orbit with eccentricity 0.9 it needed 3% to 6% more. `RK23` is only competitive for very rough answers.
* **Medium accuracy (1e-6 to 1e-9)**: `Vern8` needed the fewest diffeq calls on most problems, 7% to 29% fewer than `DOP853` for errors of 1e-6 to 1e-10. `Vern7` needed up to about 40% more calls than `DOP853` on the Lorenz, Arenstorf, Van der Pol, and oscillator problems, but up to 19% fewer on the Pleiades problem and at the loosest errors on Lorenz and Arenstorf.
* **High accuracy (1e-10 to 1e-12)**: `Vern8` and `DOP853`. `Vern7` needed 3% to 40% more calls than `DOP853` here except on the Pleiades problem. On long integrations `Vern8`'s error stops improving near a relative error of 1e-12 because rounding in its large weights accumulates. For the tightest results use `DOP853` or `Vern7`, which kept improving down to `rtol=1e-13`.
* **Expensive diffeq**: the number of diffeq calls dominates the run time, so pick the method with the fewest calls at your target accuracy (usually `Vern8`, `DOP853`, or `Vern7`). `LSODA`'s Adams formulas can need even fewer calls on smooth problems at tight tolerances, at a much higher cost per step.
* **Cheap diffeq or many dependent variables**: the solver's own work per step matters as well. The explicit methods loop over $y$ about once per stage, so their per-step overheads scale with their stage counts. In the measurements below the wall times largely follow the diffeq calls.
* **Dense output, `t_eval`, and events**: `Tsit5` and `RK45` build their interpolants for free. `DOP853`, `Vern7`, and `Vern8` make 3, 5, and 7 extra calls on each step that builds one, which with `dense_output=True` or events is every step. `DOP853` then costs 15 calls per step and `Vern8` 20, which made them about equally expensive in the measurements below and moved `Vern7` behind both. With a sparse `t_eval` only the steps containing a `t_eval` point pay this cost.
* **Interpolant accuracy**: `Vern7` and `Vern8` interpolants have the same order as the methods, so dense output is as accurate as the steps themselves. The interpolants of `RK45` and `Tsit5` are one order lower (fourth order) and `DOP853`'s is seventh order.

## Work-Precision Measurements

The tables below give the number of diffeq calls, and the wall time, that each method needed to reach a given relative error at the end of the integration. Each method was run with `rtol = atol` from 1e-3 to 1e-13. The error is not always monotone in the tolerance, so the cost of each error level is that of the cheapest run reaching it, or a log-log interpolation between the runs on the Pareto front (the runs that no cheaper run beats in error). The tables therefore compare methods at equal accuracy rather than at equal tolerance. A "-" means that the method did not reach that error within the tolerances tried (or, for `RK23`, that it was not run below `rtol=1e-10`). Errors were measured against analytic solutions (the linear oscillator and the periodic Arenstorf orbit) or against a `DOP853` solution at `rtol = atol = 1e-14`, which agreed with `Vern8` at the same tolerance to within 4e-12.

The problems are:

* **Lorenz**: $\sigma = 10$, $\rho = 28$, $\beta = 8/3$, $y_0 = (1, 0, 0)$, $t \in [0, 5]$.
* **Arenstorf**: the restricted three-body orbit of Hairer, Nørsett, and Wanner (1993) over one period, $t \in [0, 17.0652...]$. Rounding of the initial state limits every method to errors above about 3e-10.
* **Pleiades**: the seven-body problem of Hairer, Nørsett, and Wanner (1993), 28 dependent variables, $t \in [0, 3]$.
* **Van der Pol**: $\mu = 1$, $y_0 = (2, 0)$, $t \in [0, 200]$.
* **Linear oscillator**: $y'' = -y$, $y_0 = (1, 0)$, $t \in [0, 200 \pi]$.

The wall times are the fastest of 15 runs of `nbsolve2_ivp` with a compiled diffeq (CyRK 0.20.0, AMD Ryzen 7 9700X, Windows 11). They include a fixed cost of about 45 µs per call to `nbsolve2_ivp`, so the Lorenz problem, which takes little more than that, is omitted from the time tables. Differences of 10% or less are within the run-to-run noise. The script that produced these numbers is "Benchmarks/explicit_work_precision.py".

**Diffeq calls to reach a relative error**

| Lorenz | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RK23 | 2,200 | 9,900 | 46,000 | - | - |
| RK45 | 620 | 1,500 | 3,500 | 8,700 | 22,000 |
| DOP853 | 470 | 870 | 1,200 | 1,900 | 3,300 |
| Tsit5 | 570 | 1,100 | 2,600 | 6,300 | 15,000 |
| Vern7 | 480 | 830 | 1,400 | 2,500 | 4,600 |
| Vern8 | 430 | 670 | 1,100 | 1,800 | 2,900 |
| LSODA | 620 | 880 | 1,300 | 1,900 | - |

| Arenstorf | 1e-4 | 1e-6 | 1e-8 |
| --- | ---: | ---: | ---: |
| RK23 | 15,000 | - | - |
| RK45 | 1,900 | 5,300 | 14,000 |
| DOP853 | 1,600 | 2,700 | 3,600 |
| Tsit5 | 2,000 | 3,300 | 7,000 |
| Vern7 | 1,400 | 2,700 | 4,800 |
| Vern8 | 1,200 | 2,100 | 2,900 |
| LSODA | 1,400 | 2,100 | - |

| Pleiades | 1e-4 | 1e-6 | 1e-8 | 1e-10 |
| --- | ---: | ---: | ---: | ---: |
| RK23 | 4,400 | 20,000 | - | - |
| RK45 | 1,600 | 2,600 | 4,800 | 11,000 |
| DOP853 | 1,200 | 2,100 | 3,600 | 5,200 |
| Tsit5 | 1,400 | 2,400 | 4,100 | 10,000 |
| Vern7 | 1,200 | 1,800 | 2,900 | 5,100 |
| Vern8 | 1,100 | 1,800 | 2,900 | 4,300 |
| LSODA | 1,400 | 2,200 | 3,100 | - |

| Van der Pol | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RK23 | 15,000 | 46,000 | 160,000 | - | - |
| RK45 | 11,000 | 20,000 | 42,000 | 95,000 | - |
| DOP853 | 5,000 | 13,000 | 19,000 | 33,000 | 52,000 |
| Tsit5 | 9,100 | 18,000 | 36,000 | 82,000 | 190,000 |
| Vern7 | 9,000 | 14,000 | 22,000 | 34,000 | 62,000 |
| Vern8 | 5,300 | 11,000 | 15,000 | 23,000 | - |
| LSODA | 5,700 | 12,000 | 18,000 | 20,000 | - |

| Linear oscillator | 1e-4 | 1e-6 | 1e-8 | 1e-10 |
| --- | ---: | ---: | ---: | ---: |
| RK23 | 120,000 | 560,000 | - | - |
| RK45 | 17,000 | 42,000 | 110,000 | 270,000 |
| DOP853 | 6,700 | 12,000 | 21,000 | 38,000 |
| Tsit5 | 13,000 | 29,000 | 73,000 | 180,000 |
| Vern7 | 7,500 | 14,000 | 27,000 | 51,000 |
| Vern8 | 5,200 | 9,800 | 17,000 | 31,000 |
| LSODA | 7,300 | 12,000 | 15,000 | 21,000 |

**Wall time \[µs\] to reach a relative error**

| Arenstorf | 1e-4 | 1e-6 | 1e-8 |
| --- | ---: | ---: | ---: |
| RK23 | 710 | - | - |
| RK45 | 110 | 230 | 530 |
| DOP853 | 91 | 130 | 150 |
| Tsit5 | 110 | 160 | 290 |
| Vern7 | 88 | 130 | 190 |
| Vern8 | 78 | 110 | 140 |
| LSODA | 280 | 390 | - |

| Pleiades | 1e-4 | 1e-6 | 1e-8 | 1e-10 |
| --- | ---: | ---: | ---: | ---: |
| RK23 | 2,300 | 11,000 | - | - |
| RK45 | 820 | 1,300 | 2,500 | 6,000 |
| DOP853 | 630 | 1,000 | 1,800 | 3,700 |
| Tsit5 | 710 | 1,200 | 2,200 | 5,300 |
| Vern7 | 600 | 890 | 1,400 | 2,700 |
| Vern8 | 570 | 910 | 1,400 | 2,200 |
| LSODA | 1,100 | 1,700 | 2,900 | - |

| Van der Pol | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RK23 | 380 | 1,100 | 4,100 | - | - |
| RK45 | 200 | 330 | 660 | 1,500 | - |
| DOP853 | 98 | 180 | 280 | 400 | 610 |
| Tsit5 | 170 | 300 | 570 | 1,300 | 3,000 |
| Vern7 | 140 | 200 | 280 | 420 | 730 |
| Vern8 | 100 | 210 | 250 | 390 | - |
| LSODA | 390 | 760 | 1,200 | 1,400 | - |

| Linear oscillator | 1e-4 | 1e-6 | 1e-8 | 1e-10 |
| --- | ---: | ---: | ---: | ---: |
| RK23 | 2,800 | 14,000 | - | - |
| RK45 | 260 | 610 | 1,500 | 3,900 |
| DOP853 | 110 | 160 | 240 | 400 |
| Tsit5 | 210 | 430 | 1,000 | 2,800 |
| Vern7 | 110 | 180 | 300 | 890 |
| Vern8 | 97 | 130 | 190 | 330 |
| LSODA | 510 | 810 | 1,100 | 1,500 |

With an interpolant built on every step (`dense_output=True` or events), the extra stages change the comparison between the high-order methods. At a relative error of 1e-8, `DOP853`, `Vern7`, and `Vern8` needed 1,500, 2,100, and 1,600 diffeq calls on the Lorenz problem, 22,000, 31,000, and 21,000 on Van der Pol, and 27,000, 40,000, and 26,000 on the linear oscillator. `Tsit5` and `RK45` are unchanged.

## Verification

The tableaus of `Tsit5`, `Vern7`, and `Vern8` (including the extra stages and the interpolants) were checked against the Runge-Kutta order conditions in high-precision arithmetic. Every condition up to the order of each formula holds to better than 1e-33 for the source coefficients and to about 3e-32 for the 36-digit values in "rk.cpp", far below double precision. CyRK's tests check the observed convergence order of each method and its interpolant, the accuracy of dense output, `t_eval`, and events against analytic solutions, and the agreement with a tight `DOP853` solution.

## References

**Explicit Runge-Kutta methods**
* Bogacki, P. and Shampine, L. F., "A 3(2) pair of Runge-Kutta formulas", *Applied Mathematics Letters* 2 (1989) 321-325.
* Dormand, J. R. and Prince, P. J., "A family of embedded Runge-Kutta formulae", *Journal of Computational and Applied Mathematics* 6 (1980) 19-26.
* Hairer, E., Nørsett, S. P., and Wanner, G., *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd edition, Springer (1993). DOP853, its interpolant, and the Arenstorf and Pleiades problems.
* Tsitouras, Ch., "Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption", *Computers & Mathematics with Applications* 62 (2011) 770-775.
* Verner, J. H., "Numerically optimal Runge-Kutta pairs with interpolants", *Numerical Algorithms* 53 (2010) 383-396. The coefficients, including the corrected 7(6) interpolant, are on Verner's web page at Simon Fraser University.

**Implicit methods**
* See the [Implicit Methods](Implicit_Methods.md) page.

**Step size control**
* Hairer, Nørsett, and Wanner (1993), Section II.4.
* Virtanen, P. et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python", *Nature Methods* 17 (2020) 261-272.
