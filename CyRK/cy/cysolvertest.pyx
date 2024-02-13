# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.math cimport sin, cos

from CyRK.cy.cysolver cimport CySolver


cdef class CySolverTester(CySolver):

    cdef void diffeq(self) noexcept nogil:
        
        # Unpack y
        cdef double y0, y1
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]

        self.dy_ptr[0] = (1. - 0.01 * y1) * y0
        self.dy_ptr[1] = (0.02 * y0 - 1.) * y1


cdef class CySolverAccuracyTest(CySolver):

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]

        self.dy_ptr[0] = sin(self.t_now) - y1
        self.dy_ptr[1] = cos(self.t_now) + y0


cdef class CySolverExtraTest(CySolver):

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, extra_0, extra_1
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]

        extra_0 = (1. - 0.01 * y1)
        extra_1 = (0.02 * y0 - 1.)

        self.dy_ptr[0] = extra_0 * y0
        self.dy_ptr[1] = extra_1 * y1

        self.extra_output_ptr[0] = extra_0
        self.extra_output_ptr[1] = extra_1


cdef class CySolverLorenz(CySolver):

    cdef double a, b, c

    cdef void update_constants(self) noexcept nogil:

        self.a  = self.args_ptr[0]
        self.b  = self.args_ptr[1]
        self.c  = self.args_ptr[2]

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, y2
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]
        y2 = self.y_ptr[2]

        self.dy_ptr[0] = self.a * (y1 - y0)
        self.dy_ptr[1] = y0 * (self.b - y2) - y1
        self.dy_ptr[2] = y0 * y1 - self.c * y2


cdef class CySolverLorenzExtra(CySolver):

    cdef double a, b, c

    cdef void update_constants(self) noexcept nogil:

        self.a  = self.args_ptr[0]
        self.b  = self.args_ptr[1]
        self.c  = self.args_ptr[2]

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, y2, e_1, e_2, e_3
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]
        y2 = self.y_ptr[2]

        e_1 = self.a
        e_2 = (self.b - y2)
        e_3 = self.c * y2

        self.dy_ptr[0] = e_1 * (y1 - y0)
        self.dy_ptr[1] = y0 * e_2 - y1
        self.dy_ptr[2] = y0 * y1 - e_3

        self.extra_output_ptr[0] = e_1
        self.extra_output_ptr[1] = e_2
        self.extra_output_ptr[2] = e_3


cdef class CySolverLotkavolterra(CySolver):

    cdef double a, b, c, d

    cdef void update_constants(self) noexcept nogil:

        self.a = self.args_ptr[0]
        self.b = self.args_ptr[1]
        self.c = self.args_ptr[2]
        self.d = self.args_ptr[3]

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]

        self.dy_ptr[0] = self.a * y0 - self.b * y0 * y1
        self.dy_ptr[1] = -self.c * y1 + self.d * y0 * y1


cdef class CySolverPendulum(CySolver):

    cdef double coeff_1, coeff_2

    cdef void update_constants(self) noexcept nogil:

        cdef double l, m, g

        l = self.args_ptr[0]
        m = self.args_ptr[1]
        g = self.args_ptr[2]

        self.coeff_1 = (-3. * g / (2. * l))
        self.coeff_2 = (3. / (m * l**2))

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, torque
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]

        # External torque
        torque = 0.1 * sin(self.t_now)

        self.dy_ptr[0] = y1
        self.dy_ptr[1] = self.coeff_1 * sin(y0) + self.coeff_2 * torque


cdef class CySolverStiff(CySolver):

    cdef double a

    cdef void update_constants(self) noexcept nogil:

        self.a = self.args_ptr[0]

    cdef void diffeq(self) noexcept nogil:

        # Unpack y
        cdef double y0, y1, sin_, cos_
        y0 = self.y_ptr[0]
        y1 = self.y_ptr[1]

        # External torque
        sin_ = sin(self.t_now)
        cos_ = cos(self.t_now)

        self.dy_ptr[0] = -2.0 * y0 + y1 + 2 * sin_
        self.dy_ptr[1] = (self.a - 1.0) * y0 - self.a * y1 + self.a * (cos_ - sin_)
