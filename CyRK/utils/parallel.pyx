# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from cython.parallel import prange

from libc.math cimport floor  

cdef void f() noexcept nogil:
    cdef int i, j, k, x
    for i in prange(100_000):
        for j in range(i):
            for k in range(j):
                x = i - j + <int>floor(k / 2)
    

f()
        