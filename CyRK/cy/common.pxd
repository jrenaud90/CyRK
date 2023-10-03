# distutils: language = c++
from libcpp cimport bool as bool_cpp_t

from CyRK.rk.rk cimport double_numeric

cdef double SAFETY
cdef double MIN_FACTOR
cdef double MAX_FACTOR
cdef double MAX_STEP
cdef double INF
cdef double EPS
cdef double EPS_10
cdef double EPS_100
cdef Py_ssize_t MAX_INT_SIZE
cdef Py_ssize_t MAX_SIZET_SIZE
cdef double MIN_ARRAY_PREALLOCATE_SIZE
cdef double MAX_ARRAY_PREALLOCATE_SIZE
cdef double ARRAY_PREALLOC_TABS_SCALE
cdef double ARRAY_PREALLOC_RTOL_SCALE

cdef void interpolate(
        double* time_domain_full,
        double* time_domain_reduced,
        double_numeric* target_array_full,
        double_numeric* target_array_reduced,
        Py_ssize_t t_len_full,
        Py_ssize_t t_len_reduced,
        Py_ssize_t target_len,
        bool_cpp_t is_complex
        ) noexcept
