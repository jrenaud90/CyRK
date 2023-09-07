cdef extern from "interp_common.c":
    Py_ssize_t binary_search_with_guess(
            double key,
            double* array,
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
        double* desired_x_array,
        double* x_domain,
        double* dependent_values,
        double* desired_dependent_array,
        Py_ssize_t len_x,
        Py_ssize_t desired_len
        ) noexcept nogil

cpdef void interp_array(
        double[:] desired_x_array,
        double[:] x_domain,
        double[:] dependent_values,
        double[:] desired_dependent_array
        ) noexcept nogil

cdef void interp_complex_array_ptr(
        double* desired_x_array,
        double* x_domain,
        double complex* dependent_values,
        double complex* desired_dependent_array,
        Py_ssize_t len_x,
        Py_ssize_t desired_len
        ) noexcept nogil

cpdef void interp_complex_array(
        double[:] desired_x_array,
        double[:] x_domain,
        double complex[:] dependent_values,
        double complex[:] desired_dependent_array
        ) noexcept nogil