# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from CyRK.rk.rk_constants cimport \
    order_RK23, error_order_RK23, n_stages_RK23, A_rows_RK23, A_cols_RK23, \
    A_RK23, B_RK23, C_RK23, E_RK23, \
    order_RK45, error_order_RK45, n_stages_RK45, A_rows_RK45, A_cols_RK45, \
    A_RK45, B_RK45, C_RK45, E_RK45, \
    order_DOP853, error_order_DOP853, n_stages_DOP853, A_rows_DOP853, A_cols_DOP853, \
    A_DOP853, B_DOP853, C_DOP853, E3_DOP853, E5_DOP853

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
        ) noexcept nogil:

    if rk_method == 0:
        # RK23
        order[0]       = order_RK23
        error_order[0] = error_order_RK23
        n_stages[0]    = n_stages_RK23
        A_rows[0]      = A_rows_RK23
        A_cols[0]      = A_cols_RK23
        A_ptr[0]       = &A_RK23[0]
        B_ptr[0]       = &B_RK23[0]
        C_ptr[0]       = &C_RK23[0]
        E_ptr[0]       = &E_RK23[0]
    elif rk_method == 1:
        # RK45
        order[0]       = order_RK45
        error_order[0] = error_order_RK45
        n_stages[0]    = n_stages_RK45
        A_rows[0]      = A_rows_RK45
        A_cols[0]      = A_cols_RK45
        A_ptr[0]       = &A_RK45[0]
        B_ptr[0]       = &B_RK45[0]
        C_ptr[0]       = &C_RK45[0]
        E_ptr[0]       = &E_RK45[0]
    elif rk_method == 2:
        # DOP853
        order[0]       = order_DOP853
        error_order[0] = error_order_DOP853
        n_stages[0]    = n_stages_DOP853
        A_rows[0]      = A_rows_DOP853
        A_cols[0]      = A_cols_DOP853
        A_ptr[0]       = &A_DOP853[0]
        B_ptr[0]       = &B_DOP853[0]
        C_ptr[0]       = &C_DOP853[0]
        E3_ptr[0]      = &E3_DOP853[0]
        E5_ptr[0]      = &E5_DOP853[0]
    else:
        # Error: Unknown RK Method
        order[0]       = 0
        error_order[0] = 0
        n_stages[0]    = 0
        A_rows[0]      = 0
        A_cols[0]      = 0
