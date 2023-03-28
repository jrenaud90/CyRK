# RK23
cdef double RK23_C[3]
cdef double RK23_B[3]
cdef double RK23_E[4]
cdef double RK23_A[3][3]
cdef unsigned char RK23_order
cdef unsigned char RK23_error_order
cdef unsigned char RK23_n_stages
cdef unsigned char RK23_LEN_C
cdef unsigned char RK23_LEN_B
cdef unsigned char RK23_LEN_E
cdef unsigned char RK23_LEN_E3
cdef unsigned char RK23_LEN_E5
cdef unsigned char RK23_LEN_A0
cdef unsigned char RK23_LEN_A1

# RK45
cdef double RK45_C[6]
cdef double RK45_B[6]
cdef double RK45_E[7]
cdef double RK45_A[6][5]
cdef unsigned char RK45_order
cdef unsigned char RK45_error_order
cdef unsigned char RK45_n_stages
cdef unsigned char RK45_LEN_C
cdef unsigned char RK45_LEN_B
cdef unsigned char RK45_LEN_E
cdef unsigned char RK45_LEN_E3
cdef unsigned char RK45_LEN_E5
cdef unsigned char RK45_LEN_A0
cdef unsigned char RK45_LEN_A1

# DOP853
cdef double DOP_C[16]
cdef double DOP_C_REDUCED[12]
cdef double DOP_B[12]
cdef double DOP_E3[13]
cdef double DOP_E5[13]
cdef double DOP_A[16][16]
cdef double DOP_A_REDUCED[12][12]
cdef unsigned char DOP_order
cdef unsigned char DOP_error_order
cdef unsigned char DOP_n_stages
cdef unsigned char DOP_n_stages_extended
cdef unsigned char DOP_LEN_C
cdef unsigned char DOP_LEN_B
cdef unsigned char DOP_LEN_E
cdef unsigned char DOP_LEN_E3
cdef unsigned char DOP_LEN_E5
cdef unsigned char DOP_LEN_A0
cdef unsigned char DOP_LEN_A1
