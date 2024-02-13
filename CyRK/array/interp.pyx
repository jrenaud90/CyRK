# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from cython.parallel import prange

import numpy as np

from libc.math cimport isnan

# Get machine precision.
cdef double EPS
EPS = np.finfo(dtype=np.float64).eps


cdef double interp_ptr(
        double desired_x,
        double* x_domain,
        double* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for floats.

    Provided a domain, `x_domain` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `x_domain` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[float]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : float
        Desired value of `dependent_values`.

    """
    # TODO: Needs to be at least 3 item long array. Add exception here?

    cdef double left_value
    left_value = dependent_values[0]
    cdef double right_value
    right_value = dependent_values[len_x - 1]

    # Binary Search with Guess
    cdef Py_ssize_t j
    j = 0
    cdef double slope

    cdef double result
    cdef double fp_at_j
    cdef double xp_at_j
    cdef double fp_at_jp1
    cdef double xp_at_jp1

    # Perform binary search with guess
    if provided_j == -2:
        # No j provided; search for it instead.
        j = binary_search_with_guess(desired_x, x_domain, len_x, j)
    elif provided_j < -2:
        # Error
        # TODO: How to handle exception handling in a cdef function... For now just repeat the search.
        j = binary_search_with_guess(desired_x, x_domain, len_x, j)
    else:
        j = provided_j

    if j <= -1:
        result = left_value
    elif j >= len_x:
        result = right_value
    else:
        fp_at_j = dependent_values[j]
        xp_at_j = x_domain[j]
        if j == len_x - 1:
            result = fp_at_j
        elif xp_at_j == desired_x:
            result = fp_at_j
        else:
            fp_at_jp1 = dependent_values[j + 1]
            xp_at_jp1 = x_domain[j + 1]
            slope = (fp_at_jp1 - fp_at_j) / (xp_at_jp1 - xp_at_j)

            # If we get nan in one direction, try the other
            result = slope * (desired_x - xp_at_j) + fp_at_j
            if isnan(result):
                result = slope * (desired_x - xp_at_jp1) + fp_at_jp1
                if isnan(result) and (fp_at_jp1 == fp_at_j):
                    result = fp_at_j

    return result


cdef double complex interp_complex_ptr(
        double desired_x,
        double* x_domain,
        double complex* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for complex numbers.

    Provided a domain, `desired_x` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `desired_x` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[complex]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : complex
        Desired value of `dependent_values`.

    """
    # Note: Needs to be at least 3 item long array. Add exception here?

    cdef double complex left_value
    left_value = dependent_values[0]
    cdef double complex right_value
    right_value = dependent_values[len_x - 1]

    # Binary Search with Guess
    cdef Py_ssize_t j
    j = 0
    cdef double slope_real
    cdef double slope_imag
    cdef double x_slope_inverse

    cdef double result_real
    cdef double result_imag
    cdef double complex fp_at_j
    cdef double fp_at_j_real
    cdef double fp_at_j_imag
    cdef double xp_at_j
    cdef double complex fp_at_jp1
    cdef double fp_at_jp1_real
    cdef double fp_at_jp1_imag
    cdef double xp_at_jp1

    # Perform binary search with guess
    if provided_j == -2:
        # No j provided; search for it instead.
        j = binary_search_with_guess(desired_x, x_domain, len_x, j)
    elif provided_j < -2:
        # Error
        # TODO: How to handle exception handling in a cdef function... For now just repeat the search.
        j = binary_search_with_guess(desired_x, x_domain, len_x, j)
    else:
        j = provided_j

    if j <= -1:
        result_real = left_value.real
        result_imag = left_value.imag
    elif j >= len_x:
        result_real = right_value.real
        result_imag = right_value.imag
    else:
        fp_at_j = dependent_values[j]
        fp_at_j_real = fp_at_j.real
        fp_at_j_imag = fp_at_j.imag
        xp_at_j = x_domain[j]
        if j == len_x - 1:
            result_real = fp_at_j_real
            result_imag = fp_at_j_imag
        elif xp_at_j == desired_x:
            result_real = fp_at_j_real
            result_imag = fp_at_j_imag
        else:
            fp_at_jp1 = dependent_values[j + 1]
            fp_at_jp1_real = fp_at_jp1.real
            fp_at_jp1_imag = fp_at_jp1.imag
            xp_at_jp1 = x_domain[j + 1]
            x_slope_inverse = 1.0 / (xp_at_jp1 - xp_at_j)
            slope_real = (fp_at_jp1_real - fp_at_j_real) * x_slope_inverse
            slope_imag = (fp_at_jp1_imag - fp_at_j_imag) * x_slope_inverse

            # If we get nan in one direction try the other
            # Real Part
            result_real = slope_real * (desired_x - xp_at_j) + fp_at_j_real
            if isnan(result_real):
                result_real = slope_real * (desired_x - xp_at_jp1) + fp_at_jp1_real
                if isnan(result_real) and (fp_at_jp1_real == fp_at_j_real):
                    result_real = fp_at_j_real

            # Imaginary Part
            result_imag = slope_imag * (desired_x - xp_at_j) + fp_at_j_imag
            if isnan(result_imag):
                result_imag = slope_imag * (desired_x - xp_at_jp1) + fp_at_jp1_imag
                if isnan(result_imag) and (fp_at_jp1_imag == fp_at_j_imag):
                    result_imag = fp_at_j_imag

    cdef double complex result
    result = result_real + 1.0j * result_imag

    return result


cpdef double interp(
        double desired_x,
        double[:] x_domain,
        double[:] dependent_values,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for floats.

    Provided a domain, `x_domain` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `x_domain` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[float]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : float
        Desired value of `dependent_values`.

    """

    cdef Py_ssize_t len_x
    len_x = len(x_domain)

    return interp_ptr(desired_x, &x_domain[0], &dependent_values[0], len_x, provided_j)


cpdef double complex interp_complex(
        double desired_x,
        double[:] x_domain,
        double complex[:] dependent_values,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for complex numbers.

    Provided a domain, `desired_x` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `desired_x` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[complex]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : complex
        Desired value of `dependent_values`.

    """

    cdef Py_ssize_t len_x
    len_x = len(x_domain)

    return interp_complex_ptr(desired_x, &x_domain[0], &dependent_values[0], len_x, provided_j)


cdef (double, Py_ssize_t) interpj_ptr(
        double desired_x,
        double* x_domain,
        double* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for floats. This function will return the index that it found during interpolation.

    Provided a domain, `x_domain` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `x_domain` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[float]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : float
        Desired value of `dependent_values`.
    j_out : int
        The index that was found during binary_search_with_guess which can be used by other interpolation functions 
        to improve performance.

    """
    # Binary Search with Guess
    cdef Py_ssize_t j_out
    j_out = 0

    # Perform binary search with guess
    if provided_j == -2:
        # No j provided; search for it instead.
        j_out = binary_search_with_guess(desired_x, x_domain, len_x, j_out)
    elif provided_j < -2:
        # Error
        # TODO: How to handle exception handling in a cdef function... For now just repeat the search.
        j_out = binary_search_with_guess(desired_x, x_domain, len_x, j_out)
    else:
        j_out = provided_j

    return interp_ptr(desired_x, x_domain, dependent_values, len_x, provided_j=j_out), j_out


cpdef (double, Py_ssize_t) interpj(
        double desired_x,
        double[:] x_domain,
        double[:] dependent_values,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for floats. This function will return the index that it found during interpolation.

    Provided a domain, `x_domain` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `x_domain` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[float]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : float
        Desired value of `dependent_values`.
    j_out : int
        The index that was found during binary_search_with_guess which can be used by other interpolation functions 
        to improve performance.

    """
    cdef Py_ssize_t len_x
    len_x = len(x_domain)

    return interpj_ptr(desired_x, &x_domain[0], &dependent_values[0], len_x, provided_j)


cdef (double complex, Py_ssize_t) interp_complexj_ptr(
        double desired_x,
        double* x_domain,
        double complex* dependent_values,
        Py_ssize_t len_x,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for complex numbers.

    Provided a domain, `desired_x` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `desired_x` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[complex]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : complex
        Desired value of `dependent_values`.
    j_out : int
        The index that was found during binary_search_with_guess which can be used by other interpolation functions 
        to improve performance.

    """
    # Binary Search with Guess
    cdef Py_ssize_t j_out
    j_out = 0

    # Perform binary search with guess
    if provided_j == -2:
        # No j provided; search for it instead.
        j_out = binary_search_with_guess(desired_x, x_domain, len_x, j_out)
    elif provided_j < -2:
        # Error
        # TODO: How to handle exception handling in a cdef function... For now just repeat the search.
        j_out = binary_search_with_guess(desired_x, x_domain, len_x, j_out)
    else:
        j_out = provided_j

    return interp_complex_ptr(desired_x, x_domain, dependent_values, len_x, provided_j=j_out), j_out


cpdef (double complex, Py_ssize_t) interp_complexj(
        double desired_x,
        double[:] x_domain,
        double complex[:] dependent_values,
        Py_ssize_t provided_j = -2
        ) noexcept nogil:
    """ Interpolation function for complex numbers.

    Provided a domain, `desired_x` and a dependent array `dependent_values` search domain for value closest to 
    `desired_x` and return the value of `dependent_values` at that location if it is defined. Otherwise, use local 
    slopes of `desired_x` and `dependent_values` to interpolate a value of `dependent_values` at `desired_x`.

    Based on `numpy`'s `interp` function.

    Parameters
    ----------
    desired_x : float
        Location where `dependent_variables` is desired.
    x_domain : np.ndarray[float]
        Domain to search for the correct location.
    dependent_values : np.ndarray[complex]
        Dependent values that are to be returned after search and interpolation.
    provided_j : int
        Give a j index from a previous interpolation to improve performance.

    Returns
    -------
    result : complex
        Desired value of `dependent_values`.
    j_out : int
        The index that was found during binary_search_with_guess which can be used by other interpolation functions 
        to improve performance.

    """
    cdef Py_ssize_t len_x
    len_x = len(x_domain)

    return interp_complexj_ptr(desired_x, &x_domain[0], &dependent_values[0], len_x, provided_j)


cdef void interp_array_ptr(
        double* desired_x_array,
        double* x_domain,
        double* dependent_values,
        double* desired_dependent_array,
        Py_ssize_t len_x,
        Py_ssize_t desired_len
        ) noexcept nogil:

    # Array variables
    cdef Py_ssize_t index

    # Binary Search with Guess
    cdef double result

    # Since most of the use cases for this function are subsampling an array with another array, we can improve our
    #  guess by increasing it alongside the index variable. There are problems with this:
    #    1 - If the desired array is randomly distributed, rather than increasing, this will be slow.
    #    2 - The actual x domain is likely not linear. So the linear increase we are performing with this guess variable
    #        is not correct.
    cdef double x_slope
    cdef Py_ssize_t guess, j
    x_slope = <double>len_x / <double>desired_len
    if x_slope < 1.:
        x_slope = 1.

    cdef double desired_x
    for index in prange(desired_len, nogil=True):
        desired_x = desired_x_array[index]

        # Perform binary search with guess
        guess = <Py_ssize_t>x_slope * index
        j = binary_search_with_guess(desired_x, x_domain, len_x, guess)

        # Run interpolation
        result = interp_ptr(desired_x, x_domain, dependent_values, len_x, provided_j=j)

        # Store result
        desired_dependent_array[index] = result


cpdef void interp_array(
        double[:] desired_x_array,
        double[:] x_domain,
        double[:] dependent_values,
        double[:] desired_dependent_array
        ) noexcept nogil:

    # Array variables
    cdef Py_ssize_t len_x, desired_len
    len_x       = len(x_domain)
    desired_len = len(desired_x_array)

    interp_array_ptr(&desired_x_array[0], &x_domain[0], &dependent_values[0], &desired_dependent_array[0],
                     len_x, desired_len)


cdef void interp_complex_array_ptr(
        double* desired_x_array,
        double* x_domain,
        double complex* dependent_values,
        double complex* desired_dependent_array,
        Py_ssize_t len_x,
        Py_ssize_t desired_len
        ) noexcept nogil:

    # Array variables
    cdef Py_ssize_t index

    # Since most of the use cases for this function are subsampling an array with another array, we can improve our
    #  guess by increasing it alongside the index variable. There are problems with this:
    #    1 - If the desired array is randomly distributed, rather than increasing, this will be slow.
    #    2 - The actual x domain is likely not linear. So the linear increase we are performing with this guess variable
    #        is not correct.
    cdef double x_slope
    cdef Py_ssize_t guess, j
    x_slope = <double>len_x / <double>desired_len
    if x_slope < 1.:
        x_slope = 1.

    cdef double desired_x
    cdef double complex result
    for index in prange(desired_len, nogil=True):
        desired_x = desired_x_array[index]

        # Perform binary search with guess
        guess = <Py_ssize_t>x_slope * index
        j = binary_search_with_guess(desired_x, x_domain, len_x, guess)

        # Run interpolation
        result = interp_complex_ptr(desired_x, x_domain, dependent_values, len_x, provided_j=j)

        # Store result
        desired_dependent_array[index] = result


cpdef void interp_complex_array(
        double[:] desired_x_array,
        double[:] x_domain,
        double complex[:] dependent_values,
        double complex[:] desired_dependent_array
        ) noexcept nogil:

    cdef Py_ssize_t len_x, desired_len
    len_x       = len(x_domain)
    desired_len = len(desired_x_array)

    interp_complex_array_ptr(&desired_x_array[0], &x_domain[0], &dependent_values[0], &desired_dependent_array[0],
                             len_x, desired_len)
