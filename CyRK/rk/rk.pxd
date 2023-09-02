# Define fused type to handle both float and complex-valued versions of y and dydt.
ctypedef fused double_numeric:
    double
    double complex

cdef (Py_ssize_t, Py_ssize_t, Py_ssize_t, Py_ssize_t, Py_ssize_t) find_rk_properties(Py_ssize_t rk_method) noexcept nogil

cdef void populate_rk_arrays(Py_ssize_t rk_method, double_numeric* A_ptr, double_numeric* B_ptr, double* C_ptr, double_numeric* E_ptr, double_numeric* E3_ptr, double_numeric* E5_ptr) noexcept nogil
