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

cdef class CySolverLorenz(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):

        # Unpack y
        cdef double y0, y1, y2, a, b, c
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]
        y2 = self.y_new_view[2]
        a  = self.arg_array_view[0]
        b  = self.arg_array_view[1]
        c  = self.arg_array_view[2]

        self.dy_new_view[0] = a * (y1 - y0)
        self.dy_new_view[1] = y0 * (b - y2) - y1
        self.dy_new_view[2] = y0 * y1 - c * y2

cdef class CySolverLorenzExtra(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):

        # Unpack y
        cdef double y0, y1, y2, a, b, c, e_1, e_2, e_3
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]
        y2 = self.y_new_view[2]
        a  = self.arg_array_view[0]
        b  = self.arg_array_view[1]
        c  = self.arg_array_view[2]

        e_1 = a
        e_2 = (b - y2)
        e_3 = c * y2

        self.dy_new_view[0] = e_1 * (y1 - y0)
        self.dy_new_view[1] = y0 * e_2 - y1
        self.dy_new_view[2] = y0 * y1 - e_3

        self.extra_output_view[0] = e_1
        self.extra_output_view[1] = e_2
        self.extra_output_view[2] = e_3


cdef class CySolverLotkavolterra(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):

        # Unpack y
        cdef double y0, y1, a, b, c, d
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]
        a  = self.arg_array_view[0]
        b  = self.arg_array_view[1]
        c  = self.arg_array_view[2]
        d  = self.arg_array_view[3]

        self.dy_new_view[0] = a * y0 - b * y0 * y1
        self.dy_new_view[1] = -c * y1 + d * y0 * y1

cdef class CySolverPendulum(CySolver):

    @cython.exceptval(check=False)
    cdef void diffeq(self):

        # Unpack y
        cdef double y0, y1, l, m, g, torque
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]
        l  = self.arg_array_view[0]
        m  = self.arg_array_view[1]
        g  = self.arg_array_view[2]

        # External torque
        torque = 0.1 * sin(self.t_new)

        self.dy_new_view[0] = y1
        self.dy_new_view[1] = (-3. * g / (2. * l)) * sin(y0) + (3. / (m * l**2)) * torque
