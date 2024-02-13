cdef class CySolver:

    # Class attributes
    # -- Solution variables
    cdef double* solution_y_ptr
    cdef double* solution_t_ptr
    cdef double* solution_extra_ptr
    cdef double[::1] solution_t_view
    cdef double[::1] solution_y_view
    cdef double[::1] solution_extra_view

    # -- Dependent (y0) variable information
    cdef size_t y_size
    cdef double y_size_dbl, y_size_sqrt
    cdef double* y0_ptr

    # -- Time information
    cdef double t_start, t_end, t_delta, t_delta_abs, direction_inf
    cdef bint direction_flag

    # -- Optional args info
    cdef size_t num_args
    cdef double* args_ptr
    cdef bint use_args

    # -- Extra output info
    cdef bint capture_extra
    cdef size_t num_extra

    # -- Integration information
    cdef readonly char status
    cdef readonly str message
    cdef public bint success
    cdef double* tol_ptrs
    cdef double* rtols_ptr
    cdef double* atols_ptr
    cdef double first_step, max_step
    cdef bint user_provided_max_num_steps
    cdef size_t max_num_steps
    cdef size_t expected_size, current_size, num_concats
    cdef bint recalc_first_step
    cdef bint force_fail

    # -- Interpolation info
    cdef bint run_interpolation
    cdef bint interpolate_extra
    cdef size_t len_t_eval
    cdef double* t_eval_ptr

    # -- RK method information
    cdef unsigned char rk_method
    cdef size_t rk_order, error_order, rk_n_stages, rk_n_stages_plus1
    cdef double error_expo
    cdef size_t len_C, len_Arows, len_Acols
    cdef double* A_ptr
    cdef double* B_ptr
    cdef double* C_ptr
    cdef double* E_ptr
    cdef double* E3_ptr
    cdef double* E5_ptr
    # K is not constant. It is a temp storage variable used in RK calculations
    cdef double* K_ptr

    # -- Live variables
    cdef double t_now, t_old, step_size
    cdef size_t len_t, len_t_touse
    cdef double* temporary_y_ptrs
    cdef double* y_ptr
    cdef double* y_old_ptr
    cdef double* dy_ptr
    cdef double* dy_old_ptr
    cdef double* extra_output_ptrs
    cdef double* extra_output_init_ptr
    cdef double* extra_output_ptr

    # -- Pointers used during solve method
    cdef double* _solve_time_domain_array_ptr
    cdef double* _solve_y_results_array_ptr
    cdef double* _solve_extra_array_ptr

    # -- Pointers used during interpolation
    cdef double* _interpolate_solution_t_ptr
    cdef double* _interpolate_solution_y_ptr
    cdef double* _interpolate_solution_extra_ptr

    # Class functions
    cpdef void reset_state(
            self
            )

    cdef double calc_first_step(
            self
            ) noexcept nogil

    cpdef void solve(
            self,
            bint reset = *
            )

    cdef void _solve(
            self,
            bint reset = *
            )

    cdef void interpolate(
            self
            )

    cpdef void change_t_span(
            self,
            (double, double) t_span,
            bint auto_reset_state = *
            )

    cpdef void change_y0(
            self,
            const double[::1] y0,
            bint auto_reset_state = *
            )

    cdef void change_y0_pointer(
            self,
            double * y0_ptr,
            bint auto_reset_state = *
            )

    cpdef void change_args(
            self,
            tuple args,
            bint auto_reset_state = *
            )

    cpdef void change_tols(
            self,
            double rtol = *,
            double atol = *,
            const double[::1] rtols = *,
            const double[::1] atols = *,
            bint auto_reset_state = *
            )

    cpdef void change_max_step(
            self,
            double max_step,
            bint auto_reset_state = *
            )

    cpdef void change_first_step(
            self,
            double first_step,
            bint auto_reset_state = *
            )

    cpdef void change_t_eval(
            self,
            const double[::1] t_eval,
            bint auto_reset_state = *
            )

    cdef void change_t_eval_pointer(
            self,
            double* new_t_eval_ptr,
            size_t new_len_t_eval,
            bint auto_reset_state = *
            )

    cpdef void change_parameters(
            self,
            (double, double) t_span = *,
            const double[::1] y0 = *,
            tuple args = *,
            double rtol = *,
            double atol = *,
            const double[::1] rtols = *,
            const double[::1] atols = *,
            double max_step = *,
            double first_step = *,
            const double[::1] t_eval = *,
            bint auto_reset_state = *,
            bint auto_solve = *
            )

    cdef void update_constants(
            self
            ) noexcept nogil

    cdef void diffeq(
            self
            ) noexcept nogil

ctypedef void (*DiffeqType)(CySolver)

cdef extern from "rk_step.c":
    int rk_step_cf(
        # Pointer to differential equation
        DiffeqType diffeq_ptr,
        # Pointer to the CySolver instance
        CySolver cysolver_inst,

        # t-related variables
        double t_end,
        bint direction_flag,
        double direction_inf,

        # y-related variables
        size_t y_size,
        double y_size_dbl,
        double y_size_sqrt,

        # Pointers to class attributes that can change during rk_step call.
        double* t_now_ptr,
        double* y_ptr,
        double* dy_ptr,
        double* t_old_ptr,
        double* y_old_ptr,
        double* dy_old_ptr,
        double* step_size_ptr,
        char* status_ptr,

        # Integration tolerance variables and pointers
        double* atols_ptr,
        double* rtols_ptr,
        double max_step,

        # RK specific variables and pointers
        unsigned char rk_method,
        size_t rk_n_stages,
        size_t rk_n_stages_plus1,
        size_t len_Acols,
        size_t len_C,
        double* A_ptr,
        double* B_ptr,
        double* C_ptr,
        double* K_ptr,
        double* E_ptr,
        double* E3_ptr,
        double* E5_ptr,
        double error_expo,
        double min_step_factor,
        double max_step_factor,
        double error_safety
        ) noexcept nogil