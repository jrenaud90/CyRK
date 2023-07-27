# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sin, cos

from CyRK.cy.cysolver cimport CySolver


cdef class CySolverTester(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):
        
        # Unpack y
        cdef double y0, y1
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        self.dy_new_view[0] = (1. - 0.01 * y1) * y0
        self.dy_new_view[1] = (0.02 * y0 - 1.) * y1


cdef class CySolverAccuracyTest(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):

        # Unpack y
        cdef double y0, y1
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        self.dy_new_view[0] = sin(self.t_new) - y1
        self.dy_new_view[1] = cos(self.t_new) + y0


cdef class CySolverExtraTest(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):

        # Unpack y
        cdef double y0, y1, extra_0, extra_1
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        extra_0 = (1. - 0.01 * y1)
        extra_1 = (0.02 * y0 - 1.)

        self.dy_new_view[0] = extra_0 * y0
        self.dy_new_view[1] = extra_1 * y1

        self.extra_output_view[0] = extra_0
        self.extra_output_view[1] = extra_1