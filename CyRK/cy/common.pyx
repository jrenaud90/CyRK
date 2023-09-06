import cython

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from CyRK.array.interp cimport interp_array_ptr, interp_complex_array_ptr

cdef void interpolate(double* time_domain_full, double* time_domain_reduced,
                      double_numeric* target_array_full, double_numeric* target_array_reduced,
                      Py_ssize_t t_len_full, Py_ssize_t t_len_reduced, Py_ssize_t target_len,
                      bool_cpp_t is_complex):
    """ Interpolate the results of a successful integration over the user provided time domain, `time_domain_full`. """

    # Setup loop variables
    cdef Py_ssize_t i, j

    # Build a pointer array that will contain only 1 y for all ts in time_domain_full
    cdef double_numeric* array_slice_ptr
    array_slice_ptr = <double_numeric *> PyMem_Malloc(t_len_full * sizeof(double_numeric))
    if not array_slice_ptr:
        raise MemoryError()

    # Build a pointer that will store the interpolated values for 1 y at a time; size of self.len_t_eval
    cdef double_numeric* interpolated_array_slice_ptr
    interpolated_array_slice_ptr = <double_numeric *> PyMem_Malloc(t_len_reduced * sizeof(double_numeric))
    if not interpolated_array_slice_ptr:
        raise MemoryError()

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

    # Release memory of any temporary variables
    PyMem_Free(array_slice_ptr)
    PyMem_Free(interpolated_array_slice_ptr)
