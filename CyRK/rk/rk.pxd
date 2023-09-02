cdef (Py_ssize_t, Py_ssize_t, Py_ssize_t, Py_ssize_t, Py_ssize_t) find_rk_properties(Py_ssize_t rk_method) noexcept nogil

cdef void populate_rk_arrays(Py_ssize_t rk_method, double* A_ptr, double* B_ptr, double* C_ptr, double* E_ptr, double* E3_ptr, double* E5_ptr) noexcept nogil
