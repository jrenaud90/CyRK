""" Compares CyRK's implicit integrators against SciPy's, in accuracy, work, and step selection.

CyRK's BDF, LSODA, and Radau are built to reproduce SciPy's implementations. This script measures
how closely they actually do, and produces the two figures used in the "Implicit Methods"
documentation:

  1. A work-precision diagram. For each method and problem it sweeps the tolerance and plots the
     error achieved against the number of differential equation calls spent. CyRK and SciPy land on
     the same curve.
  2. A step size trace for one case where the two disagree, showing that they run bit-for-bit
     identically until rounding separates them, after which the step size controller amplifies the
     difference.

Both solvers are handed the *same* Python callable so that the derivative evaluations are
bit-identical; anything that differs is down to the solvers themselves.

Run from the "Benchmarks" directory:
    python scipy_implicit_comparison.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.integrate import solve_ivp

from CyRK import pysolve_ivp, __version__

# Where the documentation keeps its images.
IMAGE_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'Documentation', '_static', 'imgs')
VERSION_TAG = 'v' + __version__.replace('.', '-')

IMPLICIT_METHODS = ('BDF', 'LSODA', 'Radau')
# One colour per solver; the two are meant to overlap.
CYRK_STYLE = dict(color='#0072B2', marker='o', markersize=5, linewidth=1.6, linestyle='-')
SCIPY_STYLE = dict(color='#D55E00', marker='x', markersize=7, linewidth=1.2, linestyle='--')


class CallCounter:
    """ Wraps a derivative kernel and counts how many times it is called.

    The same kernel is handed to both solvers, in each one's calling convention, so that the
    arithmetic is identical and only the solvers differ.
    """

    def __init__(self, kernel, num_y):
        self.kernel = kernel
        self.num_y = num_y
        self.calls = 0

    def for_cyrk(self, dy, t, y):
        self.calls += 1
        self.kernel(dy, t, y)

    def for_scipy(self, t, y):
        self.calls += 1
        dy = np.empty(self.num_y)
        self.kernel(dy, t, y)
        return dy


def oscillator(dy, t, y):
    """Non-stiff; has the exact solution y = (-sin t, sin t + cos t) for y(0) = (0, 1)."""
    dy[0] = np.sin(t) - y[1]
    dy[1] = np.cos(t) + y[0]


def stiff_linear(dy, t, y):
    """Mildly stiff linear system."""
    dy[0] = -2000.0 * y[0] + 1000.0 * y[1]
    dy[1] = y[0] - y[1]


def robertson(dy, t, y):
    """Robertson chemical kinetics; the standard stiff benchmark."""
    dy[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2]
    dy[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1]
    dy[2] = 3.0e7 * y[1] * y[1]


PROBLEMS = (
    ('Oscillator (non-stiff)', oscillator,  (0.0, 10.0), np.asarray((0.0, 1.0))),
    ('Stiff linear',           stiff_linear, (0.0, 5.0),  np.asarray((1.0, 0.0))),
    ('Robertson (stiff)',      robertson,   (0.0, 40.0), np.asarray((1.0, 0.0, 0.0))),
)


def solve_both(kernel, time_span, y0, method, rtol, atol):
    """ Run a problem through both solvers and report work, steps, and the final state. """

    num_y = y0.size

    cyrk_counter = CallCounter(kernel, num_y)
    cyrk_result = pysolve_ivp(cyrk_counter.for_cyrk, time_span, y0, method=method,
                              rtol=rtol, atol=atol, pass_dy_as_arg=True)

    scipy_counter = CallCounter(kernel, num_y)
    scipy_result = solve_ivp(scipy_counter.for_scipy, time_span, list(y0), method=method,
                             rtol=rtol, atol=atol)

    return {
        'cyrk_calls': cyrk_counter.calls,
        'scipy_calls': scipy_counter.calls,
        'cyrk_steps': cyrk_result.steps_taken,
        'scipy_steps': len(scipy_result.t) - 1,
        'cyrk_y_end': cyrk_result.y[:, -1],
        'scipy_y_end': scipy_result.y[:, -1],
        'success': cyrk_result.success and scipy_result.success,
        }


def reference_solution(kernel, time_span, y0):
    """ A tight Radau solve, used as truth for the problems with no analytic solution. """

    result = pysolve_ivp(kernel, time_span, y0, method='Radau',
                         rtol=1.0e-13, atol=1.0e-15, pass_dy_as_arg=True)
    return result.y[:, -1]


def make_work_precision_figure(tolerances):
    """ Build the work-precision diagram and report how often the two solvers agree exactly. """

    figure, axes = plt.subplots(1, len(IMPLICIT_METHODS), figsize=(13.5, 4.4), sharey=False)
    agreement = {method: [0, 0, 0] for method in IMPLICIT_METHODS}  # identical, CyRK more, CyRK fewer

    for method_i, method in enumerate(IMPLICIT_METHODS):
        axis = axes[method_i]
        for problem_i, (problem_name, kernel, time_span, y0) in enumerate(PROBLEMS):
            truth = reference_solution(kernel, time_span, y0)

            cyrk_points = []
            scipy_points = []
            for rtol in tolerances:
                atol = rtol * 1.0e-2
                measurement = solve_both(kernel, time_span, y0, method, rtol, atol)
                if not measurement['success']:
                    continue

                cyrk_error = np.abs(measurement['cyrk_y_end'] - truth).max()
                scipy_error = np.abs(measurement['scipy_y_end'] - truth).max()
                cyrk_points.append((measurement['cyrk_calls'], max(cyrk_error, 1.0e-16)))
                scipy_points.append((measurement['scipy_calls'], max(scipy_error, 1.0e-16)))

                if measurement['cyrk_steps'] == measurement['scipy_steps']:
                    agreement[method][0] += 1
                elif measurement['cyrk_steps'] > measurement['scipy_steps']:
                    agreement[method][1] += 1
                else:
                    agreement[method][2] += 1

            cyrk_points = np.asarray(cyrk_points)
            scipy_points = np.asarray(scipy_points)
            # One line style per solver, one transparency per problem, so overlap is obvious.
            alpha = 1.0 - 0.25 * problem_i
            axis.plot(cyrk_points[:, 0], cyrk_points[:, 1], alpha=alpha,
                      label=f'CyRK - {problem_name}', **CYRK_STYLE)
            axis.plot(scipy_points[:, 0], scipy_points[:, 1], alpha=alpha,
                      label=f'SciPy - {problem_name}', **SCIPY_STYLE)

        axis.set_xscale('log')
        axis.set_yscale('log')
        # Minor tick labels collide badly on a narrow log axis; keep the ticks, drop the labels.
        axis.xaxis.set_major_locator(LogLocator(base=10.0))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel('Differential equation calls')
        if method_i == 0:
            axis.set_ylabel('Error at end of integration')
        axis.set_title(method)
        axis.grid(True, which='major', alpha=0.3)
        axis.legend(fontsize=6.5, loc='lower left')

    figure.suptitle(f'CyRK v{__version__} vs SciPy: implicit method work-precision '
                    '(CyRK circles, SciPy dashed crosses)')
    figure.tight_layout()
    output_path = os.path.join(IMAGE_DIRECTORY, f'CyRK_SciPy_Implicit_WorkPrecision_{VERSION_TAG}.png')
    figure.savefig(output_path, dpi=140)
    print(f'Wrote {output_path}')
    plt.close(figure)

    return agreement


def make_divergence_figure():
    """ Show how a rounding-level difference in one step grows through the step size controller.

    The case shown is the oscillator under BDF at rtol = 1e-8, atol = 1e-9, which is one of the few
    tolerance settings where the two solvers do not take the same number of steps.
    """

    kernel, time_span, y0 = oscillator, (0.0, 10.0), np.asarray((0.0, 1.0))
    rtol, atol = 1.0e-8, 1.0e-9

    cyrk_result = pysolve_ivp(kernel, time_span, y0, method='BDF', rtol=rtol, atol=atol,
                              pass_dy_as_arg=True)
    scipy_result = solve_ivp(lambda t, y: [np.sin(t) - y[1], np.cos(t) + y[0]],
                             time_span, list(y0), method='BDF', rtol=rtol, atol=atol)

    cyrk_steps = np.diff(cyrk_result.t)
    scipy_steps = np.diff(scipy_result.t)
    num_common = min(cyrk_steps.size, scipy_steps.size)
    relative_difference = np.abs(cyrk_steps[:num_common] - scipy_steps[:num_common]) / scipy_steps[:num_common]

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axis = axes[0]
    axis.plot(np.arange(cyrk_steps.size), cyrk_steps, markevery=8,
              label=f'CyRK ({cyrk_result.steps_taken} steps)', **CYRK_STYLE)
    axis.plot(np.arange(scipy_steps.size), scipy_steps, markevery=8,
              label=f'SciPy ({scipy_steps.size} steps)', **SCIPY_STYLE)
    axis.set_yscale('log')
    axis.set_xlabel('Step number')
    axis.set_ylabel('Step size')
    axis.set_title('Step sizes chosen (BDF, oscillator, rtol = 1e-8)')
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[1]
    # Zero differences cannot be drawn on a log axis; mark them at the bottom of the range instead.
    floor = 1.0e-17
    plotted = np.where(relative_difference > 0.0, relative_difference, floor)
    axis.plot(np.arange(num_common), plotted, color='#009E73', marker='.', markevery=4, linewidth=1.2)
    axis.axhline(np.finfo(float).eps, color='0.4', linestyle=':',
                 label='machine epsilon')
    axis.set_yscale('log')
    axis.set_ylim(floor * 0.5, 1.0)
    axis.set_xlabel('Step number')
    axis.set_ylabel('Relative difference in step size')
    axis.set_title('The two agree exactly, then rounding separates them')
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    figure.suptitle(f'CyRK v{__version__} vs SciPy: why step counts can differ')
    figure.tight_layout()
    output_path = os.path.join(IMAGE_DIRECTORY, f'CyRK_SciPy_Implicit_Divergence_{VERSION_TAG}.png')
    figure.savefig(output_path, dpi=140)
    print(f'Wrote {output_path}')
    plt.close(figure)

    first_difference = next((i for i in range(num_common) if relative_difference[i] > 0.0), None)
    return first_difference, relative_difference


if __name__ == '__main__':
    sweep_tolerances = [10.0 ** -exponent for exponent in range(4, 11)]

    agreement = make_work_precision_figure(sweep_tolerances)
    first_difference, relative_difference = make_divergence_figure()

    print('\nStep count agreement over the work-precision sweep:')
    for method, (identical, cyrk_more, cyrk_fewer) in agreement.items():
        total = identical + cyrk_more + cyrk_fewer
        print(f'  {method:6s}: {identical}/{total} identical, '
              f'CyRK took more steps in {cyrk_more} and fewer in {cyrk_fewer}')

    print(f'\nIn the BDF oscillator case the two are bit-for-bit identical for the first '
          f'{first_difference} steps.')
    print(f'The first difference is {relative_difference[first_difference]:.2e} relative, about '
          f'{relative_difference[first_difference] / np.finfo(float).eps:.0f} machine epsilon.')
