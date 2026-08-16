#include <algorithm>
#include <cmath>
#include <cstring>

#include "bdf.hpp"
#include "c_lu.hpp"
#include "dense.hpp"
#include "cysolution.hpp"

/* ========================================================================= */
/* =========================  Protected Methods  =========================== */
/* ========================================================================= */
CyrkErrorCodes BDF::p_additional_setup() noexcept
{
    this->integration_method = ODEMethod::BDF;
    this->error_exponent     = BDF_FIRST_STEP_ERROR_EXPONENT;

    // Build the method coefficients. `kappa` holds the NDF correction to the plain BDF formulas.
    const double kappa[BDF_MAX_ORDER + 1] = { 0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0 };
    this->gamma[0] = 0.0;
    for (size_t order_i = 1; order_i <= BDF_MAX_ORDER; order_i++)
    {
        this->gamma[order_i] = this->gamma[order_i - 1] + 1.0 / (double)order_i;
    }
    for (size_t order_i = 0; order_i <= BDF_MAX_ORDER; order_i++)
    {
        this->alpha[order_i]       = (1.0 - kappa[order_i]) * this->gamma[order_i];
        this->error_const[order_i] = kappa[order_i] * this->gamma[order_i] + 1.0 / (double)(order_i + 1);
    }
    // The final entry is only reached through `error_const[order + 1]`, which is never used at the
    // maximum order, but it is defined here so that the array is never read uninitialized.
    this->error_const[BDF_MAX_ORDER + 1] = 1.0 / (double)(BDF_MAX_ORDER + 2);

    const size_t l_num_y = this->num_y;

    /* BDF stores the Jacobian and its factorization as dense matrices, so its memory grows with
       the square of the number of dependent variables. Check that against the user's RAM budget
       before trying to allocate, since a problem large enough to blow past it would also be far
       too slow to factorize. */
    const double matrix_MB = 2.0 * (double)l_num_y * (double)l_num_y * (double)sizeof(double) / (1024.0 * 1024.0);
    if (matrix_MB > (double)this->storage_ptr->config_uptr->max_ram_MB) [[unlikely]]
    {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    try
    {
        this->D_vec.resize((BDF_MAX_ORDER + 3) * l_num_y);
        this->jacobian_vec.resize(l_num_y * l_num_y);
        this->lu_vec.resize(l_num_y * l_num_y);
        this->pivot_vec.resize(l_num_y);
        this->work_vec.resize(l_num_y * (5 + BDF_MAX_ORDER + 1));
        this->jac_factor_vec.resize(l_num_y);
        this->jac_work_vec.resize(this->num_dy);
    }
    catch (const std::bad_alloc&)
    {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    this->D_ptr        = this->D_vec.data();
    this->jacobian_ptr = this->jacobian_vec.data();
    this->lu_ptr       = this->lu_vec.data();
    this->pivot_ptr    = this->pivot_vec.data();

    double* const work_ptr = this->work_vec.data();
    this->y_predict_ptr = &work_ptr[0];
    this->psi_ptr       = &work_ptr[l_num_y];
    this->scale_ptr     = &work_ptr[l_num_y * 2];
    this->d_total_ptr   = &work_ptr[l_num_y * 3];
    this->newton_dy_ptr = &work_ptr[l_num_y * 4];
    this->D_temp_ptr    = &work_ptr[l_num_y * 5];

    // The history is not known until the first step size has been chosen. Zero it so that any
    // interpolator built before then simply reports the initial conditions.
    std::fill(this->D_vec.begin(), this->D_vec.end(), 0.0);

    // Seed the finite-difference perturbations; `p_estimate_jacobian` adapts them from here.
    std::fill(this->jac_factor_vec.begin(), this->jac_factor_vec.end(), std::sqrt(EPS));

    this->order              = 1;
    this->num_equal_steps    = 0;
    this->lu_is_valid        = false;
    this->jacobian_is_current = false;

    return CyrkErrorCodes::NO_ERROR;
}

CyrkErrorCodes BDF::p_finalize_setup() noexcept
{
    // Newton iterations only need to resolve the correction to well below the tolerance that the
    // error test is going to apply, so the convergence target is tied to the relative tolerance.
    double min_rtol = this->rtols_ptr[0];
    if (this->use_array_rtols)
    {
        for (size_t y_i = 1; y_i < this->num_y; y_i++)
        {
            min_rtol = std::min(min_rtol, this->rtols_ptr[y_i]);
        }
    }
    this->newton_tol = std::max(10.0 * EPS / min_rtol, std::min(0.03, std::sqrt(min_rtol)));

    // Selecting the first step size left the "now" state at a probe point. Restore it so that the
    // initial Jacobian is built at the initial conditions.
    this->t_now = this->t_start;
    std::memcpy(this->y_now_ptr, this->y_old_ptr, this->sizeof_dbl_Ny);
    std::memcpy(this->dy_now_ptr, this->dy_old_ptr, this->sizeof_dbl_Ndy);

    this->h_abs = this->step_size;

    // Seed the difference array: D[0] is the solution and D[1] is its first backward difference.
    const double h = this->h_abs * (this->direction_flag ? 1.0 : -1.0);
    std::memcpy(this->D_ptr, this->y_old_ptr, this->sizeof_dbl_Ny);
    double* const D_row_1_ptr = &this->D_ptr[this->num_y];
    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        D_row_1_ptr[y_i] = this->dy_old_ptr[y_i] * h;
    }

    this->p_update_jacobian();

    return CyrkErrorCodes::NO_ERROR;
}

void BDF::p_change_D(const size_t order_, const double factor) noexcept
{
    /* Rescale the backward differences so that they describe the same interpolating polynomial
       sampled on a grid with a step size of `factor` times the current one. Following SciPy, the
       transformation is built from two cumulative-product matrices, R (for the new spacing) and
       U (for unit spacing), and the differences are multiplied by transpose(R . U). */
    const size_t num_rows = order_ + 1;
    double R_matrix[(BDF_MAX_ORDER + 1) * (BDF_MAX_ORDER + 1)];
    double U_matrix[(BDF_MAX_ORDER + 1) * (BDF_MAX_ORDER + 1)];
    double RU_matrix[(BDF_MAX_ORDER + 1) * (BDF_MAX_ORDER + 1)];

    // Build R and U together; they only differ in the step size ratio that they are built from.
    for (size_t col_j = 0; col_j < num_rows; col_j++)
    {
        // The first row of each cumulative product is all ones.
        R_matrix[col_j] = 1.0;
        U_matrix[col_j] = 1.0;
    }
    for (size_t row_i = 1; row_i < num_rows; row_i++)
    {
        const double row_i_dbl = (double)row_i;
        // The first column below the top row is zero, so the cumulative products stay zero.
        R_matrix[row_i * num_rows] = 0.0;
        U_matrix[row_i * num_rows] = 0.0;
        for (size_t col_j = 1; col_j < num_rows; col_j++)
        {
            const double col_j_dbl = (double)col_j;
            R_matrix[row_i * num_rows + col_j] =
                R_matrix[(row_i - 1) * num_rows + col_j] * ((row_i_dbl - 1.0 - factor * col_j_dbl) / row_i_dbl);
            U_matrix[row_i * num_rows + col_j] =
                U_matrix[(row_i - 1) * num_rows + col_j] * ((row_i_dbl - 1.0 - col_j_dbl) / row_i_dbl);
        }
    }

    for (size_t row_i = 0; row_i < num_rows; row_i++)
    {
        for (size_t col_j = 0; col_j < num_rows; col_j++)
        {
            double temp_double = 0.0;
            for (size_t k = 0; k < num_rows; k++)
            {
                temp_double += R_matrix[row_i * num_rows + k] * U_matrix[k * num_rows + col_j];
            }
            RU_matrix[row_i * num_rows + col_j] = temp_double;
        }
    }

    // D_new[i] = sum_k RU[k][i] * D_old[k]
    const size_t l_num_y = this->num_y;
    for (size_t row_i = 0; row_i < num_rows; row_i++)
    {
        double* const D_temp_row_ptr = &this->D_temp_ptr[row_i * l_num_y];
        const double RU_0 = RU_matrix[row_i];  // k = 0 lives in the first row of RU
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            D_temp_row_ptr[y_i] = RU_0 * this->D_ptr[y_i];
        }
        for (size_t k = 1; k < num_rows; k++)
        {
            const double RU_k = RU_matrix[k * num_rows + row_i];
            if (RU_k == 0.0)
            {
                continue;
            }
            const double* const D_row_k_ptr = &this->D_ptr[k * l_num_y];
            for (size_t y_i = 0; y_i < l_num_y; y_i++)
            {
                D_temp_row_ptr[y_i] += RU_k * D_row_k_ptr[y_i];
            }
        }
    }
    std::memcpy(this->D_ptr, this->D_temp_ptr, sizeof(double) * num_rows * l_num_y);
}

void BDF::p_update_jacobian() noexcept
{
    if (this->jac_ptr)
    {
        this->p_call_jacobian(this->jacobian_ptr);
    }
    else
    {
        this->p_estimate_jacobian(this->jacobian_ptr);
    }
    this->jacobian_is_current = true;
    this->lu_is_valid         = false;
}

bool BDF::p_factor_iteration_matrix(const double c) noexcept
{
    const size_t l_num_y = this->num_y;

    // Build I - c * J then factorize it in place.
    for (size_t col_j = 0; col_j < l_num_y; col_j++)
    {
        double* const lu_column_ptr             = &this->lu_ptr[col_j * l_num_y];
        const double* const jacobian_column_ptr = &this->jacobian_ptr[col_j * l_num_y];
        for (size_t row_i = 0; row_i < l_num_y; row_i++)
        {
            lu_column_ptr[row_i] = -c * jacobian_column_ptr[row_i];
        }
        lu_column_ptr[col_j] += 1.0;
    }

    const size_t factor_status = c_dense_lu_factor(l_num_y, this->lu_ptr, this->pivot_ptr);
    this->lu_is_valid          = (factor_status == 0);
    this->c_of_last_factor     = c;

    return this->lu_is_valid;
}

double BDF::p_scaled_norm(const double* vector_ptr, const double* scale_ptr_) const noexcept
{
    double squared_sum = 0.0;
    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        const double scaled_value = vector_ptr[y_i] / scale_ptr_[y_i];
        squared_sum += scaled_value * scaled_value;
    }
    return std::sqrt(squared_sum) / this->num_y_sqrt;
}

bool BDF::p_solve_newton_system(const double t_new, const double c, size_t* num_iterations_ptr) noexcept
{
    /* Simplified Newton iteration for the algebraic system produced by the BDF formula. The
       iterate is kept directly in the solver's "now" state so that the differential equation can
       be called without any extra copying. */
    const size_t l_num_y = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr    = this->y_now_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr   = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_d_total_ptr  = this->d_total_ptr;
    double* const CYRK_RESTRICT l_newton_dy_ptr = this->newton_dy_ptr;

    this->t_now = t_new;
    std::memcpy(l_y_now_ptr, this->y_predict_ptr, this->sizeof_dbl_Ny);
    std::fill(l_d_total_ptr, l_d_total_ptr + l_num_y, 0.0);

    bool converged          = false;
    bool has_previous_norm  = false;
    double previous_dy_norm = 0.0;
    size_t iteration_i      = 0;

    for (iteration_i = 0; iteration_i < BDF_NEWTON_MAX_ITER; iteration_i++)
    {
        this->diffeq(this);

        // A diverging iterate can produce non-finite derivatives; give up rather than pollute the
        // solution with them.
        bool all_finite = true;
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            if (not std::isfinite(l_dy_now_ptr[y_i]))
            {
                all_finite = false;
                break;
            }
        }
        if (not all_finite) [[unlikely]]
        {
            break;
        }

        // Right hand side: c * f(t_new, y) - psi - d
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            l_newton_dy_ptr[y_i] = c * l_dy_now_ptr[y_i] - this->psi_ptr[y_i] - l_d_total_ptr[y_i];
        }
        c_dense_lu_solve(l_num_y, this->lu_ptr, this->pivot_ptr, l_newton_dy_ptr);

        const double dy_norm = this->p_scaled_norm(l_newton_dy_ptr, this->scale_ptr);

        double convergence_rate = 0.0;
        bool has_rate = false;
        if (has_previous_norm and (previous_dy_norm > 0.0))
        {
            convergence_rate = dy_norm / previous_dy_norm;
            has_rate         = true;
        }

        if (has_rate)
        {
            // Stop early if the iteration is not contracting fast enough to reach the tolerance
            // within the remaining iterations.
            const double remaining_iterations = (double)(BDF_NEWTON_MAX_ITER - iteration_i);
            if ((convergence_rate >= 1.0) or
                ((std::pow(convergence_rate, remaining_iterations) / (1.0 - convergence_rate) * dy_norm) > this->newton_tol))
            {
                break;
            }
        }

        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            l_y_now_ptr[y_i]   += l_newton_dy_ptr[y_i];
            l_d_total_ptr[y_i] += l_newton_dy_ptr[y_i];
        }

        if ((dy_norm == 0.0) or
            (has_rate and ((convergence_rate / (1.0 - convergence_rate) * dy_norm) < this->newton_tol)))
        {
            converged = true;
            break;
        }

        previous_dy_norm = dy_norm;
        has_previous_norm = true;
    }

    num_iterations_ptr[0] = iteration_i + 1;
    return converged;
}

void BDF::p_step_implementation() noexcept
{
    const size_t l_num_y  = this->num_y;
    const double direction = this->direction_flag ? 1.0 : -1.0;

    // Find the minimum step size based on the value of t (there are fewer floating point numbers
    // between neighboring values when t is large).
    const double min_step_size = 10.0 * std::abs(std::nextafter(this->t_old, this->direction_inf) - this->t_old);

    // Keep the step size within its bounds, rescaling the history to match whenever it moves.
    double h_abs_now = this->h_abs;
    if (h_abs_now > this->max_step_size)
    {
        h_abs_now = this->max_step_size;
        this->p_change_D(this->order, this->max_step_size / this->h_abs);
        this->num_equal_steps = 0;
    }
    else if (h_abs_now < min_step_size)
    {
        h_abs_now = min_step_size;
        this->p_change_D(this->order, min_step_size / this->h_abs);
        this->num_equal_steps = 0;
    }

    this->jacobian_is_current = false;

    bool step_accepted   = false;
    bool step_error      = false;
    double safety        = 0.0;
    double error_norm    = 0.0;
    double t_new_accepted = this->t_old;
    size_t l_order       = this->order;

    // !! Step Loop
    while (not step_accepted)
    {
        // This will cause integration to fail: step size smaller than spacing between numbers.
        if (h_abs_now < min_step_size) [[unlikely]]
        {
            step_error = true;
            this->storage_ptr->update_status(CyrkErrorCodes::STEP_SIZE_ERROR_SPACING);
            break;
        }

        double h     = h_abs_now * direction;
        double t_new = this->t_old + h;

        // Check that we are not at the end of integration with that move.
        if ((direction * (t_new - this->t_end)) > 0.0)
        {
            t_new = this->t_end;
            this->p_change_D(l_order, std::abs(t_new - this->t_old) / h_abs_now);
            this->num_equal_steps = 0;
            this->lu_is_valid     = false;
        }
        h              = t_new - this->t_old;
        h_abs_now      = std::abs(h);
        t_new_accepted = t_new;

        // Explicit prediction of the solution at the end of the step.
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            double predicted = this->D_ptr[y_i];
            for (size_t row_i = 1; row_i <= l_order; row_i++)
            {
                predicted += this->D_ptr[row_i * l_num_y + y_i];
            }
            this->y_predict_ptr[y_i] = predicted;
        }

        // Error weights and the history term of the algebraic system.
        double rtol = this->rtols_ptr[0];
        double atol = this->atols_ptr[0];
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            rtol = this->use_array_rtols ? this->rtols_ptr[y_i] : rtol;
            atol = this->use_array_atols ? this->atols_ptr[y_i] : atol;
            this->scale_ptr[y_i] = atol + rtol * std::abs(this->y_predict_ptr[y_i]);

            double psi_value = 0.0;
            for (size_t row_i = 1; row_i <= l_order; row_i++)
            {
                psi_value += this->D_ptr[row_i * l_num_y + y_i] * this->gamma[row_i];
            }
            this->psi_ptr[y_i] = psi_value / this->alpha[l_order];
        }

        const double c = h / this->alpha[l_order];
        bool converged = false;
        size_t num_newton_iterations = 0;
        while (not converged)
        {
            if (not this->lu_is_valid)
            {
                if (not this->p_factor_iteration_matrix(c)) [[unlikely]]
                {
                    step_error = true;
                    this->storage_ptr->update_status(CyrkErrorCodes::JACOBIAN_IS_SINGULAR);
                    break;
                }
            }

            converged = this->p_solve_newton_system(t_new, c, &num_newton_iterations);

            if (not converged)
            {
                if (this->jacobian_is_current)
                {
                    // A fresh Jacobian did not help; the step size has to come down instead.
                    break;
                }
                // Rebuild the Jacobian at the predicted state and try again.
                this->t_now = t_new;
                std::memcpy(this->y_now_ptr, this->y_predict_ptr, this->sizeof_dbl_Ny);
                this->diffeq(this);
                this->p_update_jacobian();
            }
        }

        if (step_error) [[unlikely]]
        {
            break;
        }

        if (not converged)
        {
            // Halve the step and rebuild the iteration matrix on the next attempt.
            h_abs_now *= 0.5;
            this->p_change_D(l_order, 0.5);
            this->num_equal_steps = 0;
            this->lu_is_valid     = false;
            continue;
        }

        // Newton iterations that converge quickly earn a more aggressive step size.
        safety = 0.9 * (double)(2 * BDF_NEWTON_MAX_ITER + 1) /
                 (double)(2 * BDF_NEWTON_MAX_ITER + num_newton_iterations);

        // Re-weight using the corrected solution and test the local error.
        double rtol_new = this->rtols_ptr[0];
        double atol_new = this->atols_ptr[0];
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            rtol_new = this->use_array_rtols ? this->rtols_ptr[y_i] : rtol_new;
            atol_new = this->use_array_atols ? this->atols_ptr[y_i] : atol_new;
            this->scale_ptr[y_i] = atol_new + rtol_new * std::abs(this->y_now_ptr[y_i]);
        }
        error_norm = std::abs(this->error_const[l_order]) * this->p_scaled_norm(this->d_total_ptr, this->scale_ptr);

        if (error_norm > 1.0)
        {
            const double factor = std::max(
                MIN_FACTOR,
                safety * std::pow(error_norm, -1.0 / (double)(l_order + 1)));
            h_abs_now *= factor;
            this->p_change_D(l_order, factor);
            this->num_equal_steps = 0;
            // Convergence was not the problem so the existing factorization is left alone.
        }
        else
        {
            step_accepted = true;
        }
    }

    if (step_error) [[unlikely]]
    {
        this->error_flag = true;
        return;
    }

    this->num_equal_steps++;
    this->t_now = t_new_accepted;
    this->h_abs = h_abs_now;

    /* Update the differences. The principal relation is
       D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D held the differences of the
       previous interpolating polynomial and that the total Newton correction is D^{order + 1} y_n,
       which is what makes this compact update possible. */
    double* const D_order_1_ptr = &this->D_ptr[(l_order + 1) * l_num_y];
    double* const D_order_2_ptr = &this->D_ptr[(l_order + 2) * l_num_y];
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        D_order_2_ptr[y_i] = this->d_total_ptr[y_i] - D_order_1_ptr[y_i];
        D_order_1_ptr[y_i] = this->d_total_ptr[y_i];
    }
    for (size_t row_i = l_order + 1; row_i-- > 0; )
    {
        double* const D_row_ptr      = &this->D_ptr[row_i * l_num_y];
        const double* const D_next_ptr = &this->D_ptr[(row_i + 1) * l_num_y];
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            D_row_ptr[y_i] += D_next_ptr[y_i];
        }
    }

    // Extra outputs are not part of the algebraic system, so refresh them at the accepted state.
    if (this->capture_extra)
    {
        this->diffeq(this);
    }

    /* The order is only allowed to change once the history has been built on a constant step
       size, otherwise the error estimates for the neighboring orders are not trustworthy. */
    if (this->num_equal_steps < (l_order + 1))
    {
        return;
    }

    double error_norm_minus = INF;
    if (l_order > 1)
    {
        error_norm_minus = std::abs(this->error_const[l_order - 1]) *
                           this->p_scaled_norm(&this->D_ptr[l_order * l_num_y], this->scale_ptr);
    }

    double error_norm_plus = INF;
    if (l_order < BDF_MAX_ORDER)
    {
        error_norm_plus = std::abs(this->error_const[l_order + 1]) *
                          this->p_scaled_norm(&this->D_ptr[(l_order + 2) * l_num_y], this->scale_ptr);
    }

    // Pick whichever of the neighboring orders promises the largest step size.
    const double error_norms[3] = { error_norm_minus, error_norm, error_norm_plus };
    double factors[3];
    size_t best_i = 0;
    for (size_t i = 0; i < 3; i++)
    {
        const double exponent = -1.0 / (double)(l_order + i);
        factors[i] = (error_norms[i] > 0.0) ? std::pow(error_norms[i], exponent) : INF;
        if (factors[i] > factors[best_i])
        {
            best_i = i;
        }
    }

    if (best_i == 0)
    {
        l_order -= 1;
    }
    else if (best_i == 2)
    {
        l_order += 1;
    }
    this->order = l_order;

    const double factor = std::min(MAX_FACTOR, safety * factors[best_i]);
    this->h_abs *= factor;
    this->p_change_D(l_order, factor);
    this->num_equal_steps = 0;
    this->lu_is_valid     = false;
}

/* ========================================================================= */
/* =========================  Public Methods  ============================== */
/* ========================================================================= */
/* Dense Output Methods */
void BDF::set_Q_order(size_t* Q_order_ptr)
{
    // Q holds the backward differences D[1] through D[order]; D[0] is stored separately as the
    // interpolant's base value.
    Q_order_ptr[0] = this->order;
}

void BDF::set_Q_order_max(size_t* Q_order_max_ptr)
{
    Q_order_max_ptr[0] = BDF_MAX_ORDER;
}

void BDF::set_Q_array(double* Q_ptr) noexcept
{
    const size_t l_num_y = this->num_y;
    const size_t l_order = this->order;

    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        const size_t stride_Q = y_i * l_order;
        for (size_t row_i = 1; row_i <= l_order; row_i++)
        {
            Q_ptr[stride_Q + row_i - 1] = this->D_ptr[row_i * l_num_y + y_i];
        }
    }
}

double BDF::get_dense_step() const noexcept
{
    // The history is always scaled to the step size that the solver intends to take next.
    return this->h_abs * (this->direction_flag ? 1.0 : -1.0);
}

double* BDF::get_dense_base_y_ptr() noexcept
{
    // D[0] is the solution at the end of the step.
    return this->D_ptr;
}
