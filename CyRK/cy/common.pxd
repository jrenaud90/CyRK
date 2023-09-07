# distutils: language = c++
from libcpp cimport bool as bool_cpp_t

from CyRK.rk.rk cimport double_numeric

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
