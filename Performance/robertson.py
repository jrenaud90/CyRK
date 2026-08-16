""" Robertson Differential Equations

A chemical kinetics problem describing an autocatalytic reaction between three species. The rate
constants differ by many orders of magnitude, which makes the system very stiff: the second species
reaches a quasi-steady state almost immediately while the other two evolve slowly. This is the
standard benchmark for stiff solvers, and the one problem in this set where the implicit methods
should be far ahead of the explicit ones.

.. [1] H. H. Robertson, "The solution of a set of reaction rate equations", in J. Walsh (ed.),
       Numerical Analysis: An Introduction, Academic Press, pp. 178-182, 1966.
"""
import numpy as np
from numba import njit

from CyRK import nb2cy

robertson_y0 = np.asarray((1., 0., 0.), dtype=np.float64)
robertson_args = (0.04, 1.0e4, 3.0e7)
robertson_time_span_1 = (0., 4.)
robertson_time_span_2 = (0., 40.)


@njit(cache=True)
def robertson_nb(t, y, a, b, c):

    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    dy = np.empty_like(y)
    dy[0] = -a * y0 + b * y1 * y2
    dy[1] = a * y0 - b * y1 * y2 - c * y1 * y1
    dy[2] = c * y1 * y1
    return dy


robertson_cy = nb2cy(robertson_nb, use_njit=True, cache_njit=True)
