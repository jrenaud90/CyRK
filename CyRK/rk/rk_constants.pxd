cdef size_t order_RK23
cdef size_t error_order_RK23
cdef size_t n_stages_RK23
cdef size_t A_rows_RK23
cdef size_t A_cols_RK23
cdef double[9] A_RK23
cdef double* A_RK23_ptr
cdef double[3] B_RK23
cdef double* B_RK23_ptr
cdef double[3] C_RK23
cdef double* C_RK23_ptr
cdef double[4] E_RK23
cdef double* E_RK23_ptr


cdef size_t order_RK45
cdef size_t error_order_RK45
cdef size_t n_stages_RK45
cdef size_t A_rows_RK45
cdef size_t A_cols_RK45
cdef double[30] A_RK45
cdef double* A_RK45_ptr
cdef double[6] B_RK45
cdef double* B_RK45_ptr
cdef double[6] C_RK45
cdef double* C_RK45_ptr
cdef double[7] E_RK45
cdef double* E_RK45_ptr


cdef size_t order_DOP853
cdef size_t error_order_DOP853
cdef size_t n_stages_DOP853
cdef size_t A_rows_DOP853
cdef size_t A_cols_DOP853
cdef double[144] A_DOP853
cdef double* A_DOP853_ptr
cdef double[12] B_DOP853
cdef double* B_DOP853_ptr
cdef double[12] C_DOP853
cdef double* C_DOP853_ptr
cdef double[13] E3_DOP853
cdef double* E3_DOP853_ptr
cdef double[13] E5_DOP853
cdef double* E5_DOP853_ptr
