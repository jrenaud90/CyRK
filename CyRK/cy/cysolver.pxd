cimport numpy as np
from libcpp cimport bool as bool_cpp_t
cdef class CySolver:

    # Class attributes    
    # -- Live variables
    cdef double t_new, t_old
    cdef unsigned int len_t
    cdef double[:] y_new_view, y_old_view, dy_new_view, dy_old_view
    cdef double[:] extra_output_view, extra_output_init_view
    
    # -- Dependent (y0) variable information
    cdef unsigned short y_size
    cdef double y_size_dbl, y_size_sqrt
    cdef const double[:] y0_view
    
    # -- RK method information
    cdef unsigned char rk_method
    cdef unsigned char rk_order, error_order, rk_n_stages, rk_n_stages_plus1, rk_n_stages_extended
    cdef double error_expo
    cdef unsigned char len_C
    cdef double[:] B_view, E_view, E3_view, E5_view, E_tmp_view, E3_tmp_view, E5_tmp_view, C_view
    cdef double[:, :] A_view, K_view
    
    # -- Integration information
    cdef public char status
    cdef public str message
    cdef public bool_cpp_t success
    cdef double t_start, t_end, t_delta, t_delta_abs, direction, direction_inf
    cdef double rtol, atol
    cdef double step_size, max_step
    cdef unsigned int expected_size
    cdef unsigned int num_concats
    
    # -- Optional args info
    cdef unsigned short num_args
    cdef double[:] arg_array_view

    # -- Extra output info
    cdef bool_cpp_t capture_extra
    cdef unsigned short num_extra

    # -- Interpolation info
    cdef bool_cpp_t run_interpolation
    cdef bool_cpp_t interpolate_extra
    cdef unsigned int len_t_eval
    cdef double[:] t_eval_view

    # -- Solution variables
    cdef double[:, :] solution_y_view, solution_extra_view
    cdef double[:] solution_t_view

    # Class functions
    cdef double calc_first_step(self)
    cpdef void solve(self)
    cdef void _solve(self)
    cdef void interpolate(self)
    cdef void diffeq(self)
