#include "cy_array.hpp"

size_t binary_search_with_guess(double key, const double* array, size_t length, size_t guess) {
    const size_t LIKELY_IN_CACHE_SIZE = 8;

    size_t imin = 0;
    size_t imax = length;
    size_t imid = 0;

    if (key > array[length - 1]) {
        return length;
    }
    else if (key < array[0]) {
        return -1;
    }

    if (guess > (length - 3)) {
        guess = length - 3;
    }
    if (guess < 1) {
        guess = 1;
    }

    /* Check most likely values: guess - 1, guess, guess + 1 */
    if (key < array[guess]) {
        if (key < array[guess - 1]) {
            imax = guess - 1;
            /* last attempt to restrict search to items in cache */
            if ((guess > LIKELY_IN_CACHE_SIZE) && (key >= array[guess - LIKELY_IN_CACHE_SIZE])) {
                imin = guess - LIKELY_IN_CACHE_SIZE;
            }
        }
        else {
            return guess - 1;
        }
    }
    else {
        if (key < array[guess + 1]) {
            return guess;
        }
        else {
            if (key < array[guess + 2]) {
                return guess + 1;
            }
            else {
                imin = guess + 2;
                /* last attempt to restrict search to items in cache */
                if ((guess < (length - LIKELY_IN_CACHE_SIZE - 1)) && (key < array[guess + LIKELY_IN_CACHE_SIZE])) {
                    imax = guess + LIKELY_IN_CACHE_SIZE;
                }
            }
        }
    }
    /* Finally, find index by bisection */
    while (imin < imax) {
        imid = imin + ((imax - imin) >> 1);
        if (key >= array[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }

    return imin - 1;

}