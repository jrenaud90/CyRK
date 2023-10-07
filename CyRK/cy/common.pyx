# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import cython

from libc.math cimport fmax, fmin
from libc.math cimport INFINITY as INF
from libc.float cimport DBL_EPSILON as EPS
from libc.stdint cimport SIZE_MAX, INT32_MAX
from cpython.mem cimport PyMem_Free

from CyRK.utils.utils cimport allocate_mem, reallocate_mem
from CyRK.array.interp cimport interp_array_ptr, interp_complex_array_ptr

# # Integration Constants
# Multiply steps computed from asymptotic behaviour of errors by this.
cdef double SAFETY           = 0.9
cdef double MIN_FACTOR       = 0.2  # Minimum allowed decrease in a step size.
cdef double MAX_FACTOR       = 10.  # Maximum allowed increase in a step size.
cdef double MAX_STEP         = INF
cdef double EPS_10           = EPS * 10.
cdef double EPS_100          = EPS * 100.
cdef size_t MAX_SIZET_SIZE = <size_t>(SIZE_MAX)
cdef size_t MAX_INT_SIZE   = <size_t>(INT32_MAX)

# # Memory management constants
# Assume that a cpu has a L1 of 300KB. Say that this progam will have access to 75% of that total.
cdef double CPU_CACHE_SIZE = 0.75 * 300_000.
# Number of entities we can fit into that size is based on the size of double (or double complex)
cdef double MAX_ARRAY_PREALLOCATE_SIZE_DBL      = 600_000.
cdef double MAX_ARRAY_PREALLOCATE_SIZE_DBLCMPLX = 300_000.
cdef double MIN_ARRAY_PREALLOCATE_SIZE = 10.
cdef double ARRAY_PREALLOC_TABS_SCALE  = 1000.  # A delta_t_abs higher than this value will start to grow array size.
cdef double ARRAY_PREALLOC_RTOL_SCALE  = 1.0e-5  # A rtol lower than this value will start to grow array size.


cdef void interpolate(
        double* time_domain_full,
        double* time_domain_reduced,
        double_numeric* target_array_full,
        double_numeric* target_array_reduced,
        size_t t_len_full,
        size_t t_len_reduced,
        size_t target_len,
        bool_cpp_t is_complex
        ) noexcept:
    """ Interpolate the results of a successful integration over the user provided time domain, `time_domain_full`. """

    # Setup loop variables
    cdef size_t i, j

    # Build a pointer array that will contain only 1 y for all ts in time_domain_full
    cdef double_numeric* array_slice_ptr = NULL
    array_slice_ptr = <double_numeric *> allocate_mem(
        t_len_full * sizeof(double_numeric),
        'array_slice_ptr (common.interpolate)')

    # Build a pointer that will store the interpolated values for 1 y at a time; size of self.len_t_eval
    cdef double_numeric* interpolated_array_slice_ptr = NULL
    interpolated_array_slice_ptr = <double_numeric *> allocate_mem(
        t_len_reduced * sizeof(double_numeric),
        'interpolated_array_slice_ptr (common.interpolate)')
    
    try:
        for j in range(target_len):
            # The interpolation function only works on 1D arrays, so we must loop through each of the y variables.
            # # Set timeslice equal to the time values at this y_j
            for i in range(t_len_full):
                # OPT: Inefficient memory looping
                array_slice_ptr[i] = target_array_full[i * target_len + j]

            # Perform numerical interpolation
            if double_numeric is cython.doublecomplex:
                interp_complex_array_ptr(
                    time_domain_reduced,
                    time_domain_full,
                    array_slice_ptr,
                    interpolated_array_slice_ptr,
                    t_len_full,
                    t_len_reduced)
            else:
                interp_array_ptr(
                    time_domain_reduced,
                    time_domain_full,
                    array_slice_ptr,
                    interpolated_array_slice_ptr,
                    t_len_full,
                    t_len_reduced)

            # Store result.
            for i in range(t_len_reduced):
                # OPT: Inefficient memory looping
                target_array_reduced[i * target_len + j] = interpolated_array_slice_ptr[i]
    finally:
        # Release memory of any temporary variables
        if not (array_slice_ptr is NULL):
            PyMem_Free(array_slice_ptr)
            array_slice_ptr = NULL
        if not (interpolated_array_slice_ptr is NULL):
            PyMem_Free(interpolated_array_slice_ptr)
            interpolated_array_slice_ptr = NULL

cdef size_t find_expected_size(
        size_t y_size,
        size_t num_extra,
        double t_delta_abs,
        double rtol_min,
        bool_cpp_t capture_extra,
        bool_cpp_t is_complex) noexcept nogil:

    cdef double temp_expected_size
    # Pick starting value that works with most problems
    temp_expected_size = 500.0
    # If t_delta_abs is very large or rtol is very small, then we may need more. 
    temp_expected_size = \
        fmax(
            temp_expected_size,
            fmax(
                t_delta_abs / ARRAY_PREALLOC_TABS_SCALE,
                ARRAY_PREALLOC_RTOL_SCALE / rtol_min
                )
            )
    # Fix values that are very small/large
    temp_expected_size = fmax(temp_expected_size, MIN_ARRAY_PREALLOCATE_SIZE)

    if is_complex:
        max_expected = MAX_ARRAY_PREALLOCATE_SIZE_DBL
    else:
        max_expected = MAX_ARRAY_PREALLOCATE_SIZE_DBLCMPLX
    if capture_extra:
        max_expected /= (y_size + num_extra)
    else:
        max_expected /= y_size
    
    temp_expected_size = fmin(temp_expected_size, max_expected)
    # Store result as int
    cdef size_t expected_size_to_use = <size_t>temp_expected_size
    return expected_size_to_use