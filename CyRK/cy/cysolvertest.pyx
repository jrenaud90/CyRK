# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sin, cos

from CyRK.cy.cysolver cimport CySolver


cdef class CySolverTester(CySolver):

    cdef void diffeq(self) noexcept nogil:
        
        # Unpack y
        cdef double y0, y1
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        self.dy_new_view[0] = (1. - 0.01 * y1) * y0
        self.dy_new_view[1] = (0.02 * y0 - 1.) * y1


cdef class CySolverAccuracyTest(CySolver):

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        self.dy_new_view[0] = sin(self.t_new) - y1
        self.dy_new_view[1] = cos(self.t_new) + y0


cdef class CySolverExtraTest(CySolver):

    cdef void diffeq(self) noexcept nogil:

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

    cdef double a, b, c

    cdef void update_constants(self) noexcept nogil:

        self.a  = self.arg_array_view[0]
        self.b  = self.arg_array_view[1]
        self.c  = self.arg_array_view[2]

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, y2
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]
        y2 = self.y_new_view[2]

        self.dy_new_view[0] = self.a * (y1 - y0)
        self.dy_new_view[1] = y0 * (self.b - y2) - y1
        self.dy_new_view[2] = y0 * y1 - self.c * y2


cdef class CySolverLorenzExtra(CySolver):

    cdef double a, b, c

    cdef void update_constants(self) noexcept nogil:

        self.a  = self.arg_array_view[0]
        self.b  = self.arg_array_view[1]
        self.c  = self.arg_array_view[2]

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, y2, e_1, e_2, e_3
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]
        y2 = self.y_new_view[2]

        e_1 = self.a
        e_2 = (self.b - y2)
        e_3 = self.c * y2

        self.dy_new_view[0] = e_1 * (y1 - y0)
        self.dy_new_view[1] = y0 * e_2 - y1
        self.dy_new_view[2] = y0 * y1 - e_3

        self.extra_output_view[0] = e_1
        self.extra_output_view[1] = e_2
        self.extra_output_view[2] = e_3


cdef class CySolverLotkavolterra(CySolver):

    cdef double a, b, c, d

    cdef void update_constants(self) noexcept nogil:

        self.a = self.arg_array_view[0]
        self.b = self.arg_array_view[1]
        self.c = self.arg_array_view[2]
        self.d = self.arg_array_view[3]

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        self.dy_new_view[0] = self.a * y0 - self.b * y0 * y1
        self.dy_new_view[1] = -self.c * y1 + self.d * y0 * y1


cdef class CySolverPendulum(CySolver):

    cdef double coeff_1, coeff_2

    cdef void update_constants(self) noexcept nogil:

        cdef double l, m, g

        l = self.arg_array_view[0]
        m = self.arg_array_view[1]
        g = self.arg_array_view[2]

        self.coeff_1 = (-3. * g / (2. * l))
        self.coeff_2 = (3. / (m * l**2))

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, torque
        y0 = self.y_new_view[0]
        y1 = self.y_new_view[1]

        # External torque
        torque = 0.1 * sin(self.t_new)

        self.dy_new_view[0] = y1
        self.dy_new_view[1] = self.coeff_1 * sin(y0) + self.coeff_2 * torque
