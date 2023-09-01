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

    # -- Solution variables
    cdef double[:, ::1] solution_y_view, solution_extra_view
    cdef double[::1] solution_t_view

    # -- Dependent (y0) variable information
    cdef Py_ssize_t y_size
    cdef double y_size_dbl, y_size_sqrt
    cdef double* y0_ptr

    # -- Time information
    cdef double t_start, t_end, t_delta, t_delta_abs, direction_inf
    cdef bool_cpp_t direction_flag

    # -- Optional args info
    cdef Py_ssize_t num_args
    cdef double* arg_array_ptr

    # -- Extra output info
    cdef bool_cpp_t capture_extra
    cdef Py_ssize_t num_extra

    # -- Integration information
    cdef readonly char status
    cdef readonly str message
    cdef public bool_cpp_t success
    cdef double* rtols_ptr
    cdef double* atols_ptr
    cdef double first_step, max_step
    cdef Py_ssize_t max_num_steps
    cdef Py_ssize_t expected_size, num_concats,
    cdef bool_cpp_t recalc_first_step

    # -- Interpolation info
    cdef bool_cpp_t run_interpolation
    cdef bool_cpp_t interpolate_extra
    cdef Py_ssize_t len_t_eval
    cdef const double[::1] t_eval_view

    # -- RK method information
    cdef unsigned char rk_method
    cdef Py_ssize_t rk_order, error_order, rk_n_stages, rk_n_stages_plus1, rk_n_stages_extended
    cdef double error_expo
    cdef Py_ssize_t len_C
    cdef const double[::1] B_view, E_view, E3_view, E5_view, C_view
    cdef const double[:, ::1] A_view
    # K is not constant. It is a temp storage variable used in RK calculations
    cdef double* K_ptr

    # -- Live variables
    cdef double t_new, t_old, step_size
    cdef Py_ssize_t len_t
    cdef double* y_new_ptr
    cdef double* y_old_ptr
    cdef double* dy_new_ptr
    cdef double* dy_old_ptr
    cdef double* extra_output_init_ptr
    cdef double* extra_output_ptr

    # Class functions
    cpdef void reset_state(self)

    cdef double calc_first_step(self) noexcept nogil

    cdef void rk_step(self) noexcept nogil

    cpdef void solve(self, bool_cpp_t reset = *)

    cdef void _solve(self, bool_cpp_t reset = *)

    cdef void interpolate(self)

    cpdef void change_t_span(self, (double, double) t_span, bool_cpp_t auto_reset_state = *)

    cpdef void change_y0(self, const double[::1] y0, bool_cpp_t auto_reset_state = *)

    cpdef void change_args(self, tuple args, bool_cpp_t auto_reset_state = *)

    cpdef void change_tols(self, double rtol = *,
                           double atol = *,
                           double[::1] rtols = *,
                           double[::1] atols = *,
                           bool_cpp_t auto_reset_state = *)

    cpdef void change_max_step(self, double max_step, bool_cpp_t auto_reset_state = *)

    cpdef void change_first_step(self, double first_step, bool_cpp_t auto_reset_state = *)

    cpdef void change_t_eval(self, const double[:] t_eval, bool_cpp_t auto_reset_state = *)

    cpdef void change_parameters(self, (double, double) t_span = *,
                                const double[::1] y0 = *,
                                tuple args = *,
                                double rtol = *,
                                double atol = *,
                                double[::1] rtols = *,
                                double[::1] atols = *,
                                double max_step = *,
                                double first_step = *,
                                const double[::1] t_eval = *,
                                bool_cpp_t auto_reset_state = *,
                                bool_cpp_t auto_solve = *)

    cdef void update_constants(self) noexcept nogil

    cdef void diffeq(self) noexcept nogil
