"""Work-precision benchmark of CyRK's explicit Runge-Kutta methods (and LSODA) on standard non-stiff problems.

For each problem, method, and tolerance (rtol = atol) this records the number of diffeq calls (counted by a compiled
diffeq through `pysolve_ivp`), the wall time (fastest of repeated `nbsolve2_ivp` calls with a compiled diffeq, so no
Python runs inside the integration), and the relative error at the final time against an analytic solution or a DOP853
solution at rtol = atol = 1e-14. It then prints the calls and the time each method needs to reach a set of relative
errors. The error is not always monotone in the tolerance, so the cost of a target error comes from the Pareto front of
the runs (the runs that no cheaper run beats in error), interpolated log-log between its points. These are the numbers
on the "Integration Methods" page.

Usage: python explicit_work_precision.py (also writes explicit_work_precision.json)
"""
import json
import time

import numpy as np
from numba import njit

from CyRK import nbsolve2_ivp, nb_diffeq_addr, pysolve_ivp

METHODS = ("RK23", "RK45", "DOP853", "Tsit5", "Vern7", "Vern8", "LSODA")
TOLERANCES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13)
REPEATS = 15


@njit
def lorenz(dy, t, y, args):
    dy[0] = 10.0 * (y[1] - y[0])
    dy[1] = y[0] * (28.0 - y[2]) - y[1]
    dy[2] = y[0] * y[1] - (8.0 / 3.0) * y[2]


@njit
def arenstorf(dy, t, y, args):
    mu = 0.012277471
    mu_prime = 1.0 - mu
    d1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    d2 = ((y[0] - mu_prime) ** 2 + y[1] ** 2) ** 1.5
    dy[0] = y[2]
    dy[1] = y[3]
    dy[2] = y[0] + 2.0 * y[3] - mu_prime * (y[0] + mu) / d1 - mu * (y[0] - mu_prime) / d2
    dy[3] = y[1] - 2.0 * y[2] - mu_prime * y[1] / d1 - mu * y[1] / d2


@njit
def pleiades(dy, t, y, args):
    # Hairer, Norsett, and Wanner (1993), problem PLEI: seven bodies with masses i, positions y[0:7], y[7:14] and
    # velocities y[14:21], y[21:28].
    for i in range(7):
        dy[i] = y[14 + i]
        dy[7 + i] = y[21 + i]
        ax = 0.0
        ay = 0.0
        for j in range(7):
            if j != i:
                dx = y[j] - y[i]
                dyy = y[7 + j] - y[7 + i]
                r3 = (dx * dx + dyy * dyy) ** 1.5
                ax += (j + 1.0) * dx / r3
                ay += (j + 1.0) * dyy / r3
        dy[14 + i] = ax
        dy[21 + i] = ay


@njit
def van_der_pol(dy, t, y, args):
    dy[0] = y[1]
    dy[1] = (1.0 - y[0] * y[0]) * y[1] - y[0]


@njit
def oscillator(dy, t, y, args):
    dy[0] = y[1]
    dy[1] = -y[0]


ARENSTORF_PERIOD = 17.0652165601579625588917206249
PLEIADES_Y0 = np.array([3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0,
                        3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5,
                        0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0])
PROBLEMS = {
    "Lorenz": (lorenz, (0.0, 5.0), np.array([1.0, 0.0, 0.0]), None),
    # The Arenstorf orbit is periodic, so the exact final state is the initial state.
    "Arenstorf": (arenstorf, (0.0, ARENSTORF_PERIOD), np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224]),
                  np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])),
    "Pleiades": (pleiades, (0.0, 3.0), PLEIADES_Y0, None),
    "Van der Pol (mu=1)": (van_der_pol, (0.0, 200.0), np.array([2.0, 0.0]), None),
    "Linear oscillator": (oscillator, (0.0, 200.0 * np.pi), np.array([1.0, 0.0]), np.array([1.0, 0.0])),
    }


def count_calls(diffeq, time_span, y0, method, tol):
    num_calls = np.zeros(1, dtype=np.int64)

    @njit
    def counted(dy, t, y, counter):
        counter[0] += 1
        diffeq(dy, t, y, counter)

    result = pysolve_ivp(counted, time_span, y0, method=method, rtol=tol, atol=tol, args=(num_calls,),
                         pass_dy_as_arg=True, max_num_steps=10_000_000)
    return int(num_calls[0]), result.success, int(result.steps_taken)


def main():
    out = {}
    args = np.zeros(1)
    for problem_name, (diffeq, time_span, y0, exact_end) in PROBLEMS.items():
        address = nb_diffeq_addr(diffeq)
        if exact_end is None:
            ref = nbsolve2_ivp(address, time_span, y0, method="DOP853", rtol=1e-14, atol=1e-14, args=args)
            exact_end = np.asarray(ref.y)[:, -1].copy()
            ref.free()
            check = nbsolve2_ivp(address, time_span, y0, method="Vern8", rtol=1e-14, atol=1e-14, args=args)
            reference_difference = np.max(np.abs(np.asarray(check.y)[:, -1] - exact_end)) / np.max(np.abs(exact_end))
            print(f"  reference DOP853 vs Vern8 at 1e-14: {reference_difference:.1e}")
            check.free()
        scale = np.max(np.abs(exact_end))
        print(problem_name, flush=True)
        for method in METHODS:
            for tol in TOLERANCES:
                if method == "RK23" and tol < 1e-10:
                    continue
                calls, success, steps = count_calls(diffeq, time_span, y0, method, tol)
                times = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    result = nbsolve2_ivp(address, time_span, y0, method=method, rtol=tol, atol=tol, args=args)
                    times.append(time.perf_counter() - t0)
                    y_end = np.asarray(result.y)[:, -1].copy()
                    result.free()
                error = float(np.max(np.abs(y_end - exact_end)) / scale)
                out[f"{problem_name}|{method}|{tol:.0e}"] = dict(calls=calls, steps=steps, success=bool(success),
                                                                 time_us=float(np.min(times) * 1e6), error=error)
                print(f"  {method:7s} tol {tol:.0e} calls {calls:8d} steps {steps:7d} "
                      f"time {np.min(times) * 1e6:10.1f} us error {error:.2e}", flush=True)
    with open("explicit_work_precision.json", "w", encoding="utf-8") as out_file:
        json.dump(out, out_file, indent=1)
    print_summary(out)


def cost_at_error(results, problem_name, method, key, target_error):
    """Cost (`key`) to reach `target_error`: the cheapest run that reaches it, or, between two points of the Pareto
    front of the runs, the log-log interpolation between them. None when no run reaches the target."""
    runs = sorted((value[key], value["error"]) for name, value in results.items()
                  if name.startswith(f"{problem_name}|{method}|") and value["success"] and value["error"] > 0.0)
    # Pareto front: walking up in cost, keep each run that reaches a lower error than every cheaper run.
    front = list()
    for cost, error in runs:
        if not front or error < front[-1][1]:
            front.append((cost, error))
    if not front or target_error < front[-1][1]:
        return None
    if target_error >= front[0][1]:
        return float(front[0][0])
    for (cost_0, error_0), (cost_1, error_1) in zip(front[:-1], front[1:]):
        if error_1 <= target_error <= error_0:
            fraction = np.log(target_error / error_0) / np.log(error_1 / error_0)
            return float(np.exp(np.log(cost_0) + fraction * np.log(cost_1 / cost_0)))
    return None


def print_summary(results):
    target_errors = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
    for key, label in (("calls", "Diffeq calls"), ("time_us", "Wall time [us]")):
        print()
        print(f"{label} to reach a relative error")
        for problem_name in PROBLEMS:
            print(f"{problem_name:20s}" + "".join(f"{target:>10.0e}" for target in target_errors))
            for method in METHODS:
                costs = [cost_at_error(results, problem_name, method, key, target) for target in target_errors]
                print(f"  {method:18s}" + "".join(f"{cost:10.0f}" if cost else f"{'-':>10s}" for cost in costs))


if __name__ == "__main__":
    main()
