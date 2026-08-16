#include <algorithm>
#include <cmath>
#include <cstring>

#include "radau.hpp"
#include "c_lu.hpp"
#include "dense.hpp"
#include "cysolution.hpp"

/* ========================================================================= */
/* =========================  Protected Methods  =========================== */
/* ========================================================================= */
CyrkErrorCodes RADAU::p_additional_setup() noexcept
{
    this->integration_method = ODEMethod::RADAU;
    this->error_exponent     = RADAU_FIRST_STEP_ERROR_EXPONENT;

    /* Build the method's constants. The Butcher matrix is never used directly; what the algorithm
       needs is its eigendecomposition and the transformation between the stage values and the
       eigenbasis. These values are the same ones SciPy uses. */
    const double sqrt_6 = std::sqrt(6.0);
    const double cbrt_3 = std::cbrt(3.0);
    const double cbrt_9 = cbrt_3 * cbrt_3;

    this->C[0] = (4.0 - sqrt_6) / 10.0;
    this->C[1] = (4.0 + sqrt_6) / 10.0;
    this->C[2] = 1.0;

    this->E[0] = (-13.0 - 7.0 * sqrt_6) / 3.0;
    this->E[1] = (-13.0 + 7.0 * sqrt_6) / 3.0;
    this->E[2] = -1.0 / 3.0;

    // One real eigenvalue and one complex conjugate pair of the inverse Butcher matrix.
    this->mu_real    = 3.0 + cbrt_9 - cbrt_3;
    this->mu_complex = std::complex<double>(
        3.0 + 0.5 * (cbrt_3 - cbrt_9),
        -0.5 * (std::pow(3.0, 5.0 / 6.0) + std::pow(3.0, 7.0 / 6.0)));

    const double T_values[RADAU_NUM_STAGES * RADAU_NUM_STAGES] = {
         0.09443876248897524, -0.14125529502095421,  0.03002919410514742,
         0.25021312296533332,  0.20412935229379994, -0.38294211275726192,
         1.0,                  1.0,                  0.0 };
    const double TI_values[RADAU_NUM_STAGES * RADAU_NUM_STAGES] = {
         4.17871859155190428,  0.32768282076106237,  0.52337644549944951,
        -4.17871859155190428, -0.32768282076106237,  0.47662355450055044,
         0.50287263494578682, -2.57192694985560522,  0.59603920482822492 };
    std::memcpy(this->T_matrix, T_values, sizeof(T_values));
    std::memcpy(this->TI_matrix, TI_values, sizeof(TI_values));

    // The first row of TI belongs to the real eigenvalue; the other two form the complex pair.
    for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
    {
        this->TI_real[stage_i]    = TI_values[stage_i];
        this->TI_complex[stage_i] = std::complex<double>(
            TI_values[RADAU_NUM_STAGES + stage_i],
            TI_values[2 * RADAU_NUM_STAGES + stage_i]);
    }

    const double P_values[RADAU_NUM_STAGES * RADAU_INTERPOLATOR_POWER] = {
        13.0 / 3.0 + 7.0 * sqrt_6 / 3.0, -23.0 / 3.0 - 22.0 * sqrt_6 / 3.0, 10.0 / 3.0 + 5.0 * sqrt_6,
        13.0 / 3.0 - 7.0 * sqrt_6 / 3.0, -23.0 / 3.0 + 22.0 * sqrt_6 / 3.0, 10.0 / 3.0 - 5.0 * sqrt_6,
        1.0 / 3.0,                       -8.0 / 3.0,                        10.0 / 3.0 };
    std::memcpy(this->P_matrix, P_values, sizeof(P_values));

    const size_t l_num_y = this->num_y;

    /* Radau holds the Jacobian plus a real and a complex factorization, so its memory grows with
       the square of the number of dependent variables. Check that against the user's RAM budget
       before allocating, since a problem large enough to blow past it would also be far too slow
       to factorize. */
    const double matrix_MB =
        (2.0 * (double)sizeof(double) + (double)sizeof(std::complex<double>)) *
        (double)l_num_y * (double)l_num_y / (1024.0 * 1024.0);
    if (matrix_MB > (double)this->storage_ptr->config_uptr->max_ram_MB) [[unlikely]]
    {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    try
    {
        this->jacobian_vec.resize(l_num_y * l_num_y);
        this->lu_real_vec.resize(l_num_y * l_num_y);
        this->lu_complex_vec.resize(l_num_y * l_num_y);
        this->pivot_real_vec.resize(l_num_y);
        this->pivot_complex_vec.resize(l_num_y);
        this->work_vec.resize(l_num_y * (4 * RADAU_NUM_STAGES + RADAU_INTERPOLATOR_POWER + 4));
        this->complex_work_vec.resize(l_num_y);
        this->dense_y_old_vec.resize(l_num_y);
        this->jac_factor_vec.resize(l_num_y);
        this->jac_work_vec.resize(this->num_dy);
    }
    catch (const std::bad_alloc&)
    {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    this->jacobian_ptr        = this->jacobian_vec.data();
    this->lu_real_ptr         = this->lu_real_vec.data();
    this->lu_complex_ptr      = this->lu_complex_vec.data();
    this->pivot_real_ptr      = this->pivot_real_vec.data();
    this->pivot_complex_ptr   = this->pivot_complex_vec.data();
    this->complex_rhs_ptr     = this->complex_work_vec.data();
    this->dense_y_old_ptr     = this->dense_y_old_vec.data();

    double* const work_ptr = this->work_vec.data();
    this->Z_ptr         = &work_ptr[0];
    this->Z0_ptr        = &work_ptr[l_num_y * RADAU_NUM_STAGES];
    this->W_ptr         = &work_ptr[l_num_y * RADAU_NUM_STAGES * 2];
    this->F_ptr         = &work_ptr[l_num_y * RADAU_NUM_STAGES * 3];
    this->dense_Q_ptr   = &work_ptr[l_num_y * RADAU_NUM_STAGES * 4];
    this->scale_ptr     = &work_ptr[l_num_y * (RADAU_NUM_STAGES * 4 + RADAU_INTERPOLATOR_POWER)];
    this->f_start_ptr   = &work_ptr[l_num_y * (RADAU_NUM_STAGES * 4 + RADAU_INTERPOLATOR_POWER + 1)];
    this->error_ptr     = &work_ptr[l_num_y * (RADAU_NUM_STAGES * 4 + RADAU_INTERPOLATOR_POWER + 2)];
    this->newton_dW_ptr = &work_ptr[l_num_y * (RADAU_NUM_STAGES * 4 + RADAU_INTERPOLATOR_POWER + 3)];

    std::fill(this->work_vec.begin(), this->work_vec.end(), 0.0);

    // Seed the finite-difference perturbations; `p_estimate_jacobian` adapts them from here.
    std::fill(this->jac_factor_vec.begin(), this->jac_factor_vec.end(), std::sqrt(EPS));

    this->has_dense_state     = false;
    this->has_previous_error  = false;
    this->lu_is_valid         = false;
    this->jacobian_is_current = false;

    return CyrkErrorCodes::NO_ERROR;
}

CyrkErrorCodes RADAU::p_finalize_setup() noexcept
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
    this->p_update_jacobian();

    return CyrkErrorCodes::NO_ERROR;
}

void RADAU::p_update_jacobian() noexcept
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

bool RADAU::p_factor_iteration_matrices(const double step) noexcept
{
    const size_t l_num_y = this->num_y;

    // Build mu / step * I - J for both the real eigenvalue and the complex pair.
    const double real_coefficient                = this->mu_real / step;
    const std::complex<double> complex_coefficient = this->mu_complex / step;

    for (size_t col_j = 0; col_j < l_num_y; col_j++)
    {
        double* const real_column_ptr                = &this->lu_real_ptr[col_j * l_num_y];
        std::complex<double>* const complex_column_ptr = &this->lu_complex_ptr[col_j * l_num_y];
        const double* const jacobian_column_ptr      = &this->jacobian_ptr[col_j * l_num_y];

        for (size_t row_i = 0; row_i < l_num_y; row_i++)
        {
            const double jacobian_value  = jacobian_column_ptr[row_i];
            real_column_ptr[row_i]       = -jacobian_value;
            complex_column_ptr[row_i]    = std::complex<double>(-jacobian_value, 0.0);
        }
        real_column_ptr[col_j]    += real_coefficient;
        complex_column_ptr[col_j] += complex_coefficient;
    }

    const size_t real_status    = c_dense_lu_factor(l_num_y, this->lu_real_ptr, this->pivot_real_ptr);
    const size_t complex_status = c_complex_dense_lu_factor(l_num_y, this->lu_complex_ptr, this->pivot_complex_ptr);

    this->lu_is_valid = (real_status == 0) and (complex_status == 0);
    return this->lu_is_valid;
}

double RADAU::p_scaled_norm(const double* vector_ptr, const double* scale_ptr_) const noexcept
{
    double squared_sum = 0.0;
    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        const double scaled_value = vector_ptr[y_i] / scale_ptr_[y_i];
        squared_sum += scaled_value * scaled_value;
    }
    return std::sqrt(squared_sum) / this->num_y_sqrt;
}

bool RADAU::p_solve_collocation_system(
        const double step,
        size_t* num_iterations_ptr,
        double* convergence_rate_ptr) noexcept
{
    /* Simplified Newton iteration on the three stage collocation system. Working in the eigenbasis
       of the Butcher matrix turns what would be a 3*num_y system into one real and one complex
       solve of size num_y per iteration. */
    const size_t l_num_y = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr  = this->y_now_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_Z_ptr      = this->Z_ptr;
    double* const CYRK_RESTRICT l_W_ptr      = this->W_ptr;
    double* const CYRK_RESTRICT l_F_ptr      = this->F_ptr;

    // W = TI . Z, using the starting guess for the stage increments.
    std::memcpy(l_Z_ptr, this->Z0_ptr, sizeof(double) * RADAU_NUM_STAGES * l_num_y);
    for (size_t row_i = 0; row_i < RADAU_NUM_STAGES; row_i++)
    {
        double* const W_row_ptr = &l_W_ptr[row_i * l_num_y];
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            double transformed = 0.0;
            for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
            {
                transformed += this->TI_matrix[row_i * RADAU_NUM_STAGES + stage_i] * l_Z_ptr[stage_i * l_num_y + y_i];
            }
            W_row_ptr[y_i] = transformed;
        }
    }

    const double real_coefficient                  = this->mu_real / step;
    const std::complex<double> complex_coefficient = this->mu_complex / step;

    bool converged           = false;
    bool has_previous_norm   = false;
    double previous_dW_norm  = 0.0;
    double convergence_rate  = 0.0;
    bool has_rate            = false;
    size_t iteration_i       = 0;

    for (iteration_i = 0; iteration_i < RADAU_NEWTON_MAX_ITER; iteration_i++)
    {
        // Evaluate the differential equation at each collocation point.
        bool all_finite = true;
        for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
        {
            const double* const Z_stage_ptr = &l_Z_ptr[stage_i * l_num_y];
            this->t_now = this->t_old + step * this->C[stage_i];
            for (size_t y_i = 0; y_i < l_num_y; y_i++)
            {
                l_y_now_ptr[y_i] = this->y_old_ptr[y_i] + Z_stage_ptr[y_i];
            }
            this->diffeq(this);

            double* const F_stage_ptr = &l_F_ptr[stage_i * l_num_y];
            for (size_t y_i = 0; y_i < l_num_y; y_i++)
            {
                const double derivative = l_dy_now_ptr[y_i];
                if (not std::isfinite(derivative)) [[unlikely]]
                {
                    all_finite = false;
                }
                F_stage_ptr[y_i] = derivative;
            }
        }
        if (not all_finite) [[unlikely]]
        {
            break;
        }

        /* Right hand sides in the eigenbasis:
             f_real    = TI_real . F    - mu_real / step * W_0
             f_complex = TI_complex . F - mu_complex / step * (W_1 + i W_2) */
        double* const real_rhs_ptr = this->newton_dW_ptr;
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            double real_sum = 0.0;
            std::complex<double> complex_sum(0.0, 0.0);
            for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
            {
                const double derivative = l_F_ptr[stage_i * l_num_y + y_i];
                real_sum    += this->TI_real[stage_i] * derivative;
                complex_sum += this->TI_complex[stage_i] * derivative;
            }
            real_rhs_ptr[y_i] = real_sum - real_coefficient * l_W_ptr[y_i];
            this->complex_rhs_ptr[y_i] =
                complex_sum - complex_coefficient * std::complex<double>(l_W_ptr[l_num_y + y_i], l_W_ptr[2 * l_num_y + y_i]);
        }

        c_dense_lu_solve(l_num_y, this->lu_real_ptr, this->pivot_real_ptr, real_rhs_ptr);
        c_complex_dense_lu_solve(l_num_y, this->lu_complex_ptr, this->pivot_complex_ptr, this->complex_rhs_ptr);

        // The correction to W, and its scaled norm across all three stages.
        double squared_sum = 0.0;
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            const double scale_value = this->scale_ptr[y_i];
            const double dW_0 = real_rhs_ptr[y_i];
            const double dW_1 = this->complex_rhs_ptr[y_i].real();
            const double dW_2 = this->complex_rhs_ptr[y_i].imag();
            squared_sum += (dW_0 / scale_value) * (dW_0 / scale_value)
                         + (dW_1 / scale_value) * (dW_1 / scale_value)
                         + (dW_2 / scale_value) * (dW_2 / scale_value);
        }
        const double dW_norm = std::sqrt(squared_sum) / std::sqrt((double)(RADAU_NUM_STAGES * l_num_y));

        has_rate = false;
        if (has_previous_norm and (previous_dW_norm > 0.0))
        {
            convergence_rate = dW_norm / previous_dW_norm;
            has_rate         = true;
        }

        if (has_rate)
        {
            // Stop early if the iteration is not contracting fast enough to reach the tolerance
            // within the remaining iterations.
            const double remaining_iterations = (double)(RADAU_NEWTON_MAX_ITER - iteration_i);
            if ((convergence_rate >= 1.0) or
                ((std::pow(convergence_rate, remaining_iterations) / (1.0 - convergence_rate) * dW_norm) > this->newton_tol))
            {
                break;
            }
        }

        // Apply the correction and map back to the stage increments: Z = T . W.
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            l_W_ptr[y_i]               += real_rhs_ptr[y_i];
            l_W_ptr[l_num_y + y_i]     += this->complex_rhs_ptr[y_i].real();
            l_W_ptr[2 * l_num_y + y_i] += this->complex_rhs_ptr[y_i].imag();
        }
        for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
        {
            double* const Z_stage_ptr = &l_Z_ptr[stage_i * l_num_y];
            for (size_t y_i = 0; y_i < l_num_y; y_i++)
            {
                double transformed = 0.0;
                for (size_t col_j = 0; col_j < RADAU_NUM_STAGES; col_j++)
                {
                    transformed += this->T_matrix[stage_i * RADAU_NUM_STAGES + col_j] * l_W_ptr[col_j * l_num_y + y_i];
                }
                Z_stage_ptr[y_i] = transformed;
            }
        }

        if ((dW_norm == 0.0) or
            (has_rate and ((convergence_rate / (1.0 - convergence_rate) * dW_norm) < this->newton_tol)))
        {
            converged = true;
            break;
        }

        previous_dW_norm  = dW_norm;
        has_previous_norm = true;
    }

    num_iterations_ptr[0]   = iteration_i + 1;
    convergence_rate_ptr[0] = has_rate ? convergence_rate : 0.0;
    return converged;
}

void RADAU::p_build_dense_state(const double step) noexcept
{
    // Q = Z^T . P, which are the coefficients of the collocation polynomial over this step.
    const size_t l_num_y = this->num_y;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        for (size_t power_i = 0; power_i < RADAU_INTERPOLATOR_POWER; power_i++)
        {
            double temp_double = 0.0;
            for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
            {
                temp_double += this->Z_ptr[stage_i * l_num_y + y_i] * this->P_matrix[stage_i * RADAU_INTERPOLATOR_POWER + power_i];
            }
            this->dense_Q_ptr[y_i * RADAU_INTERPOLATOR_POWER + power_i] = temp_double;
        }
    }
    this->dense_t_old     = this->t_old;
    this->dense_step      = step;
    this->has_dense_state = true;
}

double RADAU::p_predict_step_factor(const double h_abs_now, const double error_norm) const noexcept
{
    /* Step size prediction from Hairer and Wanner, Sec. IV.8. When the previous step is available
       the two-step formula is used, which damps oscillations in the step size. */
    if (error_norm == 0.0) [[unlikely]]
    {
        return MAX_FACTOR;
    }

    double multiplier = 1.0;
    if (this->has_previous_error and (this->h_abs_previous > 0.0) and (this->error_norm_previous > 0.0))
    {
        multiplier = (h_abs_now / this->h_abs_previous) * std::pow(this->error_norm_previous / error_norm, 0.25);
    }

    return std::min(1.0, multiplier) * std::pow(error_norm, -0.25);
}

void RADAU::p_step_implementation() noexcept
{
    const size_t l_num_y   = this->num_y;
    const double direction = this->direction_flag ? 1.0 : -1.0;

    // Find the minimum step size based on the value of t (there are fewer floating point numbers
    // between neighboring values when t is large).
    const double min_step_size = 10.0 * std::abs(std::nextafter(this->t_old, this->direction_inf) - this->t_old);

    // The derivative at the start of the step is needed by the error estimate.
    std::memcpy(this->f_start_ptr, this->dy_old_ptr, sizeof(double) * l_num_y);

    // The two-step size predictor compares against the step that was proposed coming in, which is
    // not necessarily the one this step ends up taking.
    const double h_abs_at_entry = this->h_abs;
    double h_abs_now = this->h_abs;
    if (h_abs_now > this->max_step_size)
    {
        h_abs_now = this->max_step_size;
        this->has_previous_error = false;
    }
    else if (h_abs_now < min_step_size)
    {
        h_abs_now = min_step_size;
        this->has_previous_error = false;
    }

    bool step_accepted    = false;
    bool step_error       = false;
    bool step_rejected    = false;
    double safety         = 0.0;
    double error_norm     = 0.0;
    double convergence_rate = 0.0;
    double t_new_accepted = this->t_old;
    double step_accepted_size = 0.0;
    size_t num_newton_iterations = 0;

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

        double step  = h_abs_now * direction;
        double t_new = this->t_old + step;

        // Check that we are not at the end of integration with that move.
        if ((direction * (t_new - this->t_end)) > 0.0)
        {
            t_new = this->t_end;
        }
        step      = t_new - this->t_old;
        h_abs_now = std::abs(step);

        /* Starting guess for the stage increments. The collocation polynomial of the previous step
           extrapolates well onto this one; without it the iteration starts from zero. */
        if (this->has_dense_state and (this->dense_step != 0.0))
        {
            for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
            {
                const double stage_time = this->t_old + step * this->C[stage_i];
                const double step_factor = (stage_time - this->dense_t_old) / this->dense_step;
                double powers[RADAU_INTERPOLATOR_POWER];
                double cumulative_prod = 1.0;
                for (size_t power_i = 0; power_i < RADAU_INTERPOLATOR_POWER; power_i++)
                {
                    cumulative_prod *= step_factor;
                    powers[power_i] = cumulative_prod;
                }

                double* const Z0_stage_ptr = &this->Z0_ptr[stage_i * l_num_y];
                for (size_t y_i = 0; y_i < l_num_y; y_i++)
                {
                    const double* const Q_row_ptr = &this->dense_Q_ptr[y_i * RADAU_INTERPOLATOR_POWER];
                    double interpolated = 0.0;
                    for (size_t power_i = 0; power_i < RADAU_INTERPOLATOR_POWER; power_i++)
                    {
                        interpolated += Q_row_ptr[power_i] * powers[power_i];
                    }
                    // The interpolant is an offset from the start of the step it was built over.
                    Z0_stage_ptr[y_i] = (this->dense_y_old_ptr[y_i] + interpolated) - this->y_old_ptr[y_i];
                }
            }
        }
        else
        {
            std::fill(this->Z0_ptr, this->Z0_ptr + RADAU_NUM_STAGES * l_num_y, 0.0);
        }

        // Error weights for the Newton convergence test.
        double rtol = this->rtols_ptr[0];
        double atol = this->atols_ptr[0];
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            rtol = this->use_array_rtols ? this->rtols_ptr[y_i] : rtol;
            atol = this->use_array_atols ? this->atols_ptr[y_i] : atol;
            this->scale_ptr[y_i] = atol + std::abs(this->y_old_ptr[y_i]) * rtol;
        }

        bool converged = false;
        while (not converged)
        {
            if (not this->lu_is_valid)
            {
                if (not this->p_factor_iteration_matrices(step)) [[unlikely]]
                {
                    step_error = true;
                    this->storage_ptr->update_status(CyrkErrorCodes::JACOBIAN_IS_SINGULAR);
                    break;
                }
            }

            converged = this->p_solve_collocation_system(step, &num_newton_iterations, &convergence_rate);

            if (not converged)
            {
                if (this->jacobian_is_current)
                {
                    // A fresh Jacobian did not help; the step size has to come down instead.
                    break;
                }
                // Rebuild the Jacobian at the start of the step and try again.
                this->t_now = this->t_old;
                std::memcpy(this->y_now_ptr, this->y_old_ptr, this->sizeof_dbl_Ny);
                std::memcpy(this->dy_now_ptr, this->dy_old_ptr, this->sizeof_dbl_Ndy);
                this->p_update_jacobian();
            }
        }

        if (step_error) [[unlikely]]
        {
            break;
        }

        if (not converged)
        {
            // Halve the step and rebuild the iteration matrices on the next attempt.
            h_abs_now *= 0.5;
            this->lu_is_valid = false;
            continue;
        }

        /* The last stage lands on the end of the step, so it carries the new solution. The error is
           estimated with the embedded third order formula and then filtered through the real
           iteration matrix, which is what makes the estimate usable on a stiff problem. */
        const double* const Z_last_ptr = &this->Z_ptr[(RADAU_NUM_STAGES - 1) * l_num_y];
        double rtol_new = this->rtols_ptr[0];
        double atol_new = this->atols_ptr[0];
        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            double stage_combination = 0.0;
            for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
            {
                stage_combination += this->Z_ptr[stage_i * l_num_y + y_i] * this->E[stage_i];
            }
            this->error_ptr[y_i] = this->f_start_ptr[y_i] + stage_combination / step;

            const double y_new_value = this->y_old_ptr[y_i] + Z_last_ptr[y_i];
            rtol_new = this->use_array_rtols ? this->rtols_ptr[y_i] : rtol_new;
            atol_new = this->use_array_atols ? this->atols_ptr[y_i] : atol_new;
            this->scale_ptr[y_i] =
                atol_new + std::max(std::abs(this->y_old_ptr[y_i]), std::abs(y_new_value)) * rtol_new;
        }
        c_dense_lu_solve(l_num_y, this->lu_real_ptr, this->pivot_real_ptr, this->error_ptr);
        error_norm = this->p_scaled_norm(this->error_ptr, this->scale_ptr);

        // Newton iterations that converge quickly earn a more aggressive step size.
        safety = 0.9 * (double)(2 * RADAU_NEWTON_MAX_ITER + 1) /
                 (double)(2 * RADAU_NEWTON_MAX_ITER + num_newton_iterations);

        if (step_rejected and (error_norm > 1.0))
        {
            /* A second rejection in a row usually means the error estimate itself is too crude.
               Re-filter it about the perturbed state, which gives a more reliable value. */
            this->t_now = this->t_old;
            for (size_t y_i = 0; y_i < l_num_y; y_i++)
            {
                this->y_now_ptr[y_i] = this->y_old_ptr[y_i] + this->error_ptr[y_i];
            }
            this->diffeq(this);

            for (size_t y_i = 0; y_i < l_num_y; y_i++)
            {
                double stage_combination = 0.0;
                for (size_t stage_i = 0; stage_i < RADAU_NUM_STAGES; stage_i++)
                {
                    stage_combination += this->Z_ptr[stage_i * l_num_y + y_i] * this->E[stage_i];
                }
                this->error_ptr[y_i] = this->dy_now_ptr[y_i] + stage_combination / step;
            }
            c_dense_lu_solve(l_num_y, this->lu_real_ptr, this->pivot_real_ptr, this->error_ptr);
            error_norm = this->p_scaled_norm(this->error_ptr, this->scale_ptr);
        }

        if (error_norm > 1.0)
        {
            const double factor = this->p_predict_step_factor(h_abs_now, error_norm);
            h_abs_now *= std::max(MIN_FACTOR, safety * factor);
            this->lu_is_valid = false;
            step_rejected     = true;
        }
        else
        {
            step_accepted      = true;
            t_new_accepted     = t_new;
            step_accepted_size = step;
        }
    }

    if (step_error) [[unlikely]]
    {
        this->error_flag = true;
        return;
    }

    // Advance the state to the end of the step.
    const double* const Z_last_ptr = &this->Z_ptr[(RADAU_NUM_STAGES - 1) * l_num_y];
    this->t_now = t_new_accepted;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        this->y_now_ptr[y_i] = this->y_old_ptr[y_i] + Z_last_ptr[y_i];
    }

    // Build the interpolator before `y_old` is moved on; it is also the next step's starting guess.
    std::memcpy(this->dense_y_old_vec.data(), this->y_old_ptr, this->sizeof_dbl_Ny);
    this->p_build_dense_state(step_accepted_size);

    /* A step that needed several Newton iterations without converging quickly is a sign that the
       Jacobian has gone stale, so refresh it at the new state. This also leaves `dy_now` holding
       the derivative at the accepted state, which is what the next step needs. */
    const bool recompute_jacobian = (num_newton_iterations > 2) and (convergence_rate > RADAU_JACOBIAN_REFRESH_RATE);

    double step_factor = std::min(MAX_FACTOR, safety * this->p_predict_step_factor(h_abs_now, error_norm));
    if ((not recompute_jacobian) and (step_factor < RADAU_MIN_STEP_CHANGE_FACTOR))
    {
        // Not worth refactorizing the iteration matrices for a change this small.
        step_factor = 1.0;
    }
    else
    {
        this->lu_is_valid = false;
    }

    // The derivative at the new state is needed by the next step's error estimate (and by any
    // extra output the user asked for).
    this->diffeq(this);

    if (recompute_jacobian)
    {
        this->p_update_jacobian();
    }
    else
    {
        this->jacobian_is_current = false;
    }

    this->h_abs_previous       = h_abs_at_entry;
    this->error_norm_previous  = error_norm;
    this->has_previous_error   = true;
    this->h_abs                = h_abs_now * step_factor;
}

/* ========================================================================= */
/* =========================  Public Methods  ============================== */
/* ========================================================================= */
/* Dense Output Methods */
void RADAU::set_Q_order(size_t* Q_order_ptr)
{
    // The collocation polynomial is cubic, so Q holds three coefficient columns per variable.
    Q_order_ptr[0] = RADAU_INTERPOLATOR_POWER;
}

void RADAU::set_Q_array(double* Q_ptr) noexcept
{
    std::memcpy(Q_ptr, this->dense_Q_ptr, sizeof(double) * this->num_y * RADAU_INTERPOLATOR_POWER);
}
