from libcpp cimport bool as bool_cpp_t

cdef double SAFETY
cdef double MIN_FACTOR
cdef double MAX_FACTOR
cdef double MAX_STEP
cdef double INF
cdef double EPS
cdef double EPS_10
cdef double EPS_100

cdef class CySolver:

    # Class attributes    
    # -- Live variables
    cdef double t_new, t_old
    cdef Py_ssize_t len_t
    cdef double[:] y_new_view, y_old_view, dy_new_view, dy_old_view
    cdef double[:] extra_output_view, extra_output_init_view
    
    # -- Dependent (y0) variable information
    cdef Py_ssize_t y_size
    cdef double y_size_dbl, y_size_sqrt
    cdef const double[:] y0_view
    
    # -- RK method information
    cdef unsigned char rk_method
    cdef unsigned char rk_order, error_order, rk_n_stages, rk_n_stages_plus1, rk_n_stages_extended
    cdef double error_expo
    cdef Py_ssize_t len_C
    cdef double[:] B_view, E_view, E3_view, E5_view, E_tmp_view, E3_tmp_view, E5_tmp_view, C_view
    cdef double[:, :] A_view, K_view
    
    # -- Integration information
    cdef public char status
    cdef public str message
    cdef public bool_cpp_t success
    cdef double t_start, t_end, t_delta, t_delta_abs, direction_inf
    cdef bool_cpp_t direction_flag
    cdef double rtol, atol
    cdef double step_size, max_step
    cdef double first_step
    cdef Py_ssize_t expected_size
    cdef unsigned int num_concats
    cdef double[:] scale_view
    cdef bool_cpp_t recalc_firststep
    
    # -- Optional args info
    cdef Py_ssize_t num_args
    cdef double[:] arg_array_view

    # -- Extra output info
    cdef bool_cpp_t capture_extra
    cdef Py_ssize_t num_extra

    # -- Interpolation info
    cdef bool_cpp_t run_interpolation
    cdef bool_cpp_t interpolate_extra
    cdef Py_ssize_t len_t_eval
    cdef double[:] t_eval_view

    # -- Solution variables
    cdef double[:, :] solution_y_view, solution_extra_view
    cdef double[:] solution_t_view

    # Class functions
    cpdef void reset_state(self)
    cdef double calc_first_step(self)
    cpdef void solve(self, bool_cpp_t reset = *)
    cdef void _solve(self, bool_cpp_t reset = *)
    cdef void interpolate(self)
    cdef void diffeq(self)
    cpdef void change_t_span(self, (double, double) t_span, bool_cpp_t auto_reset_state = *)
    cpdef void change_y0(self, const double[:] y0, bool_cpp_t auto_reset_state = *)
    cpdef void change_args(self, tuple args, bool_cpp_t auto_reset_state = *)
    cpdef void change_tols(self, double rtol = *, double atol = *, bool_cpp_t auto_reset_state = *)
    cpdef void change_max_step(self, double max_step, bool_cpp_t auto_reset_state = *)
    cpdef void change_first_step(self, double first_step, bool_cpp_t auto_reset_state = *)
    cpdef void change_t_eval(self, const double[:] t_eval, bool_cpp_t auto_reset_state = *)
    cpdef void change_parameters(self, (double, double) t_span = *, const double[:] y0 = *, tuple args = *,
                                double rtol = *, double atol = *, double max_step = *, double first_step = *,
                                const double[:] t_eval = *, bool_cpp_t auto_reset_state = *, bool_cpp_t auto_solve = *)
