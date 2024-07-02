# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.math cimport sin, cos


cdef void baseline_diffeq(double* dy_ptr, double t, double* y_ptr, double* args_ptr) noexcept nogil:
    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = (1. - 0.01 * y1) * y0
    dy_ptr[1] = (0.02 * y0 - 1.) * y1


cdef void accuracy_test_diffeq(double* dy_ptr, double t, double* y_ptr, double* args_ptr) noexcept nogil:
    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = sin(t) - y1
    dy_ptr[1] = cos(t) + y0


cdef void extraoutput_test_diffeq(double* dy_ptr, double t, double* y_ptr, double* args_ptr) noexcept nogil:
    # Unpack y
    cdef double y0, y1, extra_0, extra_1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    extra_0 = (1. - 0.01 * y1)
    extra_1 = (0.02 * y0 - 1.)

    # Store dy/dt
    dy_ptr[0] = extra_0 * y0
    dy_ptr[1] = extra_1 * y1

    # Store extra output
    dy_ptr[2] = extra_0
    dy_ptr[3] = extra_1


cdef void lorenz_diffeq(double* dy_ptr, double t, double* y_ptr, double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double a = args_ptr[0]
    cdef double b = args_ptr[1]
    cdef double c = args_ptr[2]

    # Unpack y
    cdef double y0, y1, y2
    y0 = y_ptr[0]
    y1 = y_ptr[1]
    y2 = y_ptr[2]

    dy_ptr[0] = a * (y1 - y0)
    dy_ptr[1] = y0 * (b - y2) - y1
    dy_ptr[2] = y0 * y1 - c * y2


cdef void lorenz_extraoutput_diffeq(double* dy_ptr, double t, double* y_ptr, double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double a = args_ptr[0]
    cdef double b = args_ptr[1]
    cdef double c = args_ptr[2]

    # Unpack y
    cdef double y0, y1, y2
    y0 = y_ptr[0]
    y1 = y_ptr[1]
    y2 = y_ptr[2]

    cdef double e_1 = a
    cdef double e_2 = (b - y2)
    cdef double e_3 = c * y2

    dy_ptr[0] = e_1 * (y1 - y0)
    dy_ptr[1] = y0 * e_2 - y1
    dy_ptr[2] = y0 * y1 - e_3

    dy_ptr[3] = e_1
    dy_ptr[4] = e_2
    dy_ptr[5] = e_3


cdef void lotkavolterra_diffeq(double* dy_ptr, double t, double* y_ptr, double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double a = args_ptr[0]
    cdef double b = args_ptr[1]
    cdef double c = args_ptr[2]
    cdef double d = args_ptr[3]

    # Unpack y
    cdef double y0, y1
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    dy_ptr[0] = a * y0 - b * y0 * y1
    dy_ptr[1] = -c * y1 + d * y0 * y1



cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, double* args_ptr) noexcept nogil:
    # Unpack args
    cdef double l = args_ptr[0]
    cdef double m = args_ptr[1]
    cdef double g = args_ptr[2]

    cdef double coeff_1 = (-3. * g / (2. * l))
    cdef double coeff_2 = (3. / (m * l**2))

    # Unpack y
    cdef double y0, y1, torque
    y0 = y_ptr[0]
    y1 = y_ptr[1]

    # External torque
    torque = 0.1 * sin(t)

    dy_ptr[0] = y1
    dy_ptr[1] = coeff_1 * sin(y0) + coeff_2 * torque