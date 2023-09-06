# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import cython
from cython.parallel import parallel, prange
cimport cython

cimport openmp

import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport isnan

# Get machine precision.
cdef double EPS
EPS = np.finfo(dtype=np.float64).eps

# Determine cache limits.
cdef Py_ssize_t LIKELY_IN_CACHE_SIZE = 8


cdef Py_ssize_t binary_search_with_guess(double key, double[:] array, Py_ssize_t length, Py_ssize_t guess) noexcept nogil:
    """ Binary search with guess.

    Based on `numpy`'s `binary_search_with_guess` function.

    Parameters
    ----------
    key : float
        Key index to search for.
    array : np.ndarray
        Array to search in.
    length : int (unsigned)
        Length of array.
    guess : int (unsigned)
        Initial guess of where key might be.

    Returns
    -------
    guess : int (unsigned)
        Corrected guess after search.
    """

    cdef Py_ssize_t imin = 0
    cdef Py_ssize_t imax = length

    if key > array[length - 1]:
        return length
    elif key < array[0]:
        return -1

    # If len <= 4 use linear search.
    # if length <= 4:
    #     raise NotImplemented

    if guess > (length - 3):
        guess = length - 3
    if guess < 1:
        guess = 1

    # Check most likely values: guess - 1, guess, guess + 1
    if key < array[guess]:
        if key < array[guess - 1]:
            imax = guess - 1
            # last attempt to restrict search to items in cache
            if guess > LIKELY_IN_CACHE_SIZE and key >= array[guess - LIKELY_IN_CACHE_SIZE]:
                imin = guess - LIKELY_IN_CACHE_SIZE

        else:
            return guess - 1
    else:
        if key < array[guess + 1]:
            return guess
        else:
            if key < array[guess + 2]:
                return guess + 1
            else:
                imin = guess + 2
                # last attempt to restrict search to items in cache
                if guess < (length - LIKELY_IN_CACHE_SIZE - 1) and key < array[guess + LIKELY_IN_CACHE_SIZE]:
                    imax = guess + LIKELY_IN_CACHE_SIZE

    # Finally, find index by bisection
    cdef Py_ssize_t imid
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if key >= array[imid]:
            imin = imid + 1
        else:
            imax = imid

    return imin - 1

cdef Py_ssize_t binary_search_with_guess_ptr(double key, double* array, Py_ssize_t length, Py_ssize_t guess) noexcept nogil:
    """ Binary search with guess.

    Based on `numpy`'s `binary_search_with_guess` function.

    Parameters
    ----------
    key : float
        Key index to search for.
    array : np.ndarray
        Array to search in.
    length : int (unsigned)
        Length of array.
    guess : int (unsigned)
        Initial guess of where key might be.

    Returns
    -------
    guess : int (unsigned)
        Corrected guess after search.
    """

    cdef Py_ssize_t imin = 0
    cdef Py_ssize_t imax = length

    if key > array[length - 1]:
        return length
    elif key < array[0]:
        return -1

    # If len <= 4 use linear search.
    # if length <= 4:
    #     raise NotImplemented

    if guess > (length - 3):
        guess = length - 3
    if guess < 1:
        guess = 1

    # Check most likely values: guess - 1, guess, guess + 1
    if key < array[guess]:
        if key < array[guess - 1]:
            imax = guess - 1
            # last attempt to restrict search to items in cache
            if guess > LIKELY_IN_CACHE_SIZE and key >= array[guess - LIKELY_IN_CACHE_SIZE]:
                imin = guess - LIKELY_IN_CACHE_SIZE

        else:
            return guess - 1
    else:
        if key < array[guess + 1]:
            return guess
        else:
            if key < array[guess + 2]:
                return guess + 1
            else:
                imin = guess + 2
                # last attempt to restrict search to items in cache
                if guess < (length - LIKELY_IN_CACHE_SIZE - 1) and key < array[guess + LIKELY_IN_CACHE_SIZE]:
                    imax = guess + LIKELY_IN_CACHE_SIZE

    # Finally, find index by bisection
    cdef Py_ssize_t imid
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if key >= array[imid]:
            imin = imid + 1
        else:
            imax = imid

    return imin - 1

cpdef (double, Py_ssize_t) interpj(double desired_x, double[:] x_domain, double[:] dependent_values,
                                    Py_ssize_t provided_j = -2) noexcept nogil:
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

    cdef Py_ssize_t j_out
    j_out = j

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

    return result, j_out


cpdef (double complex, Py_ssize_t) interp_complexj(double desired_x, double[:] x_domain,
                                                    double complex[:] dependent_values, Py_ssize_t provided_j = -2) noexcept nogil:
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

    cdef Py_ssize_t j_out
    j_out =  j

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

    return result, j_out


cpdef double interp(double desired_x, double[:] x_domain, double[:] dependent_values, Py_ssize_t provided_j = -2) noexcept nogil:
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


cpdef double complex interp_complex(double desired_x, double[:] x_domain,
                                    double complex[:] dependent_values, Py_ssize_t provided_j = -2) noexcept nogil:
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


cpdef void interp_array(double[:] desired_x_array, double[:] x_domain, double[:] dependent_values,
                        double[:] desired_dependent_array) noexcept nogil:

    # Array variables
    cdef Py_ssize_t index 
    cdef Py_ssize_t desired_len
    desired_len = len(desired_x_array)

    # Interpolation variables
    cdef Py_ssize_t len_x
    len_x = len(x_domain)
    cdef double left_value
    left_value = dependent_values[0]
    cdef double right_value
    right_value = dependent_values[len_x - 1]

    # Binary Search with Guess
    cdef Py_ssize_t j
    cdef double slope
    cdef double result
    cdef double fp_at_j
    cdef double xp_at_j
    cdef double fp_at_jp1
    cdef double xp_at_jp1

    # Since most of the use cases for this function are subsampling an array with another array, we can improve our
    #  guess by increasing it alongside the index variable. There are problems with this:
    #    1 - If the desired array is randomly distributed, rather than increasing, this will be slow.
    #    2 - The actual x domain is likely not linear. So the linear increase we are performing with this guess variable
    #        is not correct.
    cdef double x_slope
    cdef Py_ssize_t guess
    x_slope = <double>len_x / <double>desired_len
    if x_slope < 1.:
        x_slope = 1.

    cdef double desired_x
    for index in prange(desired_len, nogil=True):
        desired_x = desired_x_array[index]

        # Perform binary search with guess
        guess = <Py_ssize_t>x_slope * index
        j = binary_search_with_guess(desired_x, x_domain, len_x, guess)

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

        desired_dependent_array[index] = result

cdef void interp_array_ptr(double* desired_x_array, double* x_domain, double* dependent_values,
                           double* desired_dependent_array, Py_ssize_t len_x, Py_ssize_t desired_len) noexcept nogil:

    # Array variables
    cdef Py_ssize_t index

    # Interpolation variables
    cdef double left_value
    left_value = dependent_values[0]
    cdef double right_value
    right_value = dependent_values[len_x - 1]

    # Binary Search with Guess
    cdef Py_ssize_t j
    cdef double slope
    cdef double result
    cdef double fp_at_j
    cdef double xp_at_j
    cdef double fp_at_jp1
    cdef double xp_at_jp1

    # Since most of the use cases for this function are subsampling an array with another array, we can improve our
    #  guess by increasing it alongside the index variable. There are problems with this:
    #    1 - If the desired array is randomly distributed, rather than increasing, this will be slow.
    #    2 - The actual x domain is likely not linear. So the linear increase we are performing with this guess variable
    #        is not correct.
    cdef double x_slope
    cdef Py_ssize_t guess
    x_slope = <double>len_x / <double>desired_len
    if x_slope < 1.:
        x_slope = 1.

    cdef double desired_x
    for index in prange(desired_len, nogil=True):
        desired_x = desired_x_array[index]

        # Perform binary search with guess
        guess = <Py_ssize_t>x_slope * index
        j = binary_search_with_guess_ptr(desired_x, x_domain, len_x, guess)

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

        desired_dependent_array[index] = result


cpdef void interp_complex_array(double[:] desired_x_array, double[:] x_domain, double complex[:] dependent_values,
                                double complex[:] desired_dependent_array) noexcept nogil:

    # Array variables
    cdef Py_ssize_t index 
    cdef Py_ssize_t desired_len
    desired_len = len(desired_x_array)


    cdef Py_ssize_t len_x
    len_x = len(x_domain)
    # Note: Needs to be at least 3 item long array. Add exception here?

    # Interpolation variables
    cdef double complex left_value
    left_value = dependent_values[0]
    cdef double complex right_value
    right_value = dependent_values[len_x - 1]

    # Binary Search with Guess
    cdef Py_ssize_t j
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

    # Since most of the use cases for this function are subsampling an array with another array, we can improve our
    #  guess by increasing it alongside the index variable. There are problems with this:
    #    1 - If the desired array is randomly distributed, rather than increasing, this will be slow.
    #    2 - The actual x domain is likely not linear. So the linear increase we are performing with this guess variable
    #        is not correct.
    cdef double x_slope
    cdef Py_ssize_t guess
    x_slope = <double>len_x / <double>desired_len
    if x_slope < 1.:
        x_slope = 1.

    cdef double desired_x
    cdef double complex result
    for index in prange(desired_len, nogil=True):
        desired_x = desired_x_array[index]

        # Perform binary search with guess
        guess = <Py_ssize_t>x_slope * index
        # Perform binary search with guess
        j = binary_search_with_guess(desired_x, x_domain, len_x, guess)

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

        result = result_real + 1.0j * result_imag

        desired_dependent_array[index] = result

cdef void interp_complex_array_ptr(double* desired_x_array, double* x_domain, double complex* dependent_values,
                                   double complex* desired_dependent_array, Py_ssize_t len_x, Py_ssize_t desired_len) noexcept nogil:

    # Array variables
    cdef Py_ssize_t index

    # Note: Needs to be at least 3 item long array. Add exception here?

    # Interpolation variables
    cdef double complex left_value
    left_value = dependent_values[0]
    cdef double complex right_value
    right_value = dependent_values[len_x - 1]

    # Binary Search with Guess
    cdef Py_ssize_t j
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

    # Since most of the use cases for this function are subsampling an array with another array, we can improve our
    #  guess by increasing it alongside the index variable. There are problems with this:
    #    1 - If the desired array is randomly distributed, rather than increasing, this will be slow.
    #    2 - The actual x domain is likely not linear. So the linear increase we are performing with this guess variable
    #        is not correct.
    cdef double x_slope
    cdef Py_ssize_t guess
    x_slope = <double>len_x / <double>desired_len
    if x_slope < 1.:
        x_slope = 1.

    cdef double desired_x
    cdef double complex result
    for index in prange(desired_len, nogil=True):
        desired_x = desired_x_array[index]

        # Perform binary search with guess
        guess = <Py_ssize_t>x_slope * index
        # Perform binary search with guess
        j = binary_search_with_guess_ptr(desired_x, x_domain, len_x, guess)

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

        result = result_real + 1.0j * result_imag

        desired_dependent_array[index] = result