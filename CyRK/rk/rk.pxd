cdef void find_rk_properties(
    unsigned char rk_method,
    Py_ssize_t* order,
    Py_ssize_t* error_order,
    Py_ssize_t* n_stages,
    Py_ssize_t* A_rows,
    Py_ssize_t* A_cols,
    double** A_ptr,
    double** B_ptr,
    double** C_ptr,
    double** E_ptr,
    double** E3_ptr,
    double** E5_ptr
    ) noexcept nogil
