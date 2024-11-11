cdef extern from "interp_common.c":
    Py_ssize_t c_binary_search_with_guess(
            double key,
            const double* array,
            Py_ssize_t length,
            Py_ssize_t guess
            ) noexcept nogil

cdef double interp_ptr(
        double desired_x,
        double* x_domain,
        double* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cdef double complex interp_complex_ptr(
        double desired_x,
        double* x_domain,
        double complex* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cpdef double interp(
        double desired_x,
        double[:] x_domain,
        double[:] dependent_values,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cpdef double complex interp_complex(
        double desired_x,
        double[:] x_domain,
        double complex[:] dependent_values,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cdef (double, Py_ssize_t) interpj_ptr(
        double desired_x,
        double* x_domain,
        double* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cpdef (double, Py_ssize_t) interpj(
        double desired_x,
        double[:] x_domain,
        double[:] dependent_values,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cdef (double complex, Py_ssize_t) interp_complexj_ptr(
        double desired_x,
        double* x_domain,
        double complex* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cpdef (double complex, Py_ssize_t) interp_complexj(
        double desired_x,
        double[:] x_domain,
        double complex[:] dependent_values,
        Py_ssize_t provided_j = *
        ) noexcept nogil

cdef void interp_array_ptr(
        const double* desired_x_array,
        const double* x_domain,
        const double* dependent_values,
        double* desired_dependent_array,
        Py_ssize_t len_x,
        Py_ssize_t desired_len
        ) noexcept nogil

cpdef void interp_array(
        const double[::1] desired_x_array,
        const double[::1] x_domain,
        const double[::1] dependent_values,
        double[::1] desired_dependent_array
        ) noexcept nogil

cdef void interp_complex_array_ptr(
        const double* desired_x_array,
        const double* x_domain,
        const double complex* dependent_values,
        double complex* desired_dependent_array,
        Py_ssize_t len_x,
        Py_ssize_t desired_len
        ) noexcept nogil

cpdef void interp_complex_array(
        const double[::1] desired_x_array,
        const double[::1] x_domain,
        const double complex[::1] dependent_values,
        double complex[::1] desired_dependent_array
        ) noexcept nogil