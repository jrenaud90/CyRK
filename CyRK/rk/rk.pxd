import numpy as np
cimport numpy as np
np.import_array()

# RK23
cdef const double[::1] RK23_C_view, RK23_B_view, RK23_E_view
cdef const double[:, ::1] RK23_A_view
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
cdef const double[::1] RK45_C_view, RK45_B_view, RK45_E_view
cdef const double[:, ::1] RK45_A_view
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
cdef const double[::1] DOP_C_view, DOP_C_REDUCED_view, DOP_B_view, DOP_E3_view, DOP_E5_view
cdef const double[:, ::1] DOP_A_view, DOP_A_REDUCED_view
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
