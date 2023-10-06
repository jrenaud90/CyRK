cdef void find_rk_properties(
    unsigned char rk_method,
    size_t* order,
    size_t* error_order,
    size_t* n_stages,
    size_t* A_rows,
    size_t* A_cols,
    double** A_ptr,
    double** B_ptr,
    double** C_ptr,
    double** E_ptr,
    double** E3_ptr,
    double** E5_ptr
    ) noexcept nogil
