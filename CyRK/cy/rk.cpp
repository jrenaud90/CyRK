#include <stdexcept>
#include <numeric>
#include <algorithm>

#include "rk.hpp"
#include "dense.hpp"
#include "cysolution.hpp"

// ########################################################################################################################
// RKConfig
// ########################################################################################################################
RKConfig::RKConfig(
        DiffeqFuncType diffeq_ptr_,
        double t_start_,
        double t_end_,
        std::vector<double>& y0_vec_,
        std::vector<char>& args_vec_,
        std::vector<double>& t_eval_vec_,
        size_t num_extra_,
        size_t expected_size_,
        size_t max_num_steps_,
        size_t max_ram_MB_,
        PreEvalFunc pre_eval_func_,
        bool capture_dense_output_,
        bool force_retain_solver_,
        std::vector<Event>& events_vec_,
        std::vector<double>& rtols_,
        std::vector<double>& atols_,
        double max_step_size_,
        double first_step_size_):
            rtols(rtols_),
            atols(atols_),
            max_step_size(max_step_size_),
            first_step_size(first_step_size_),
            ProblemConfig(
                diffeq_ptr_,
                t_start_,
                t_end_,
                y0_vec_,
                args_vec_,
                t_eval_vec_,
                num_extra_,
                expected_size_,
                max_num_steps_,
                max_ram_MB_,
                pre_eval_func_,
                capture_dense_output_,
                force_retain_solver_,
                events_vec_)
{
}

void RKConfig::update_properties(
        DiffeqFuncType diffeq_ptr_,
        double t_start_,
        double t_end_,
        std::vector<double>& y0_vec_,
        std::vector<char>& args_vec_,
        std::vector<double>& t_eval_vec_,
        size_t num_extra_,
        size_t expected_size_,
        size_t max_num_steps_,
        size_t max_ram_MB_,
        PreEvalFunc pre_eval_func_,
        bool capture_dense_output_,
        bool force_retain_solver_,
        std::vector<Event>& events_vec_,
        std::vector<double>& rtols_,
        std::vector<double>& atols_,
        double max_step_size_,
        double first_step_size_)
{
    this->rtols           = rtols_;
    this->atols           = atols_;
    this->max_step_size   = max_step_size_;
    this->first_step_size = first_step_size_;

    ProblemConfig::update_properties(
        diffeq_ptr_,
        t_start_,
        t_end_,
        y0_vec_,
        args_vec_,
        t_eval_vec_,
        num_extra_,
        expected_size_,
        max_num_steps_,
        max_ram_MB_,
        pre_eval_func_,
        capture_dense_output_,
        force_retain_solver_,
        events_vec_
        );
}

void RKConfig::initialize()
{
    ProblemConfig::initialize();
    this->initialized = false;
    if ((this->rtols.size() != 1) and (this->rtols.size() != this->num_y))
    {
        throw std::length_error("Unexpected size of rtols; must be the same as y_vec or 1.");
    }
    if ((this->atols.size() != 1) and (this->atols.size() != this->num_y))
    {
        throw std::length_error("Unexpected size of rtols; must be the same as y_vec or 1.");
    }
    this->initialized = true;
}

void RKConfig::update_properties_from_config(RKConfig* new_config_ptr)
{
    this->update_properties(
        new_config_ptr->diffeq_ptr,
        new_config_ptr->t_start,
        new_config_ptr->t_end,
        new_config_ptr->y0_vec,
        new_config_ptr->args_vec,
        new_config_ptr->t_eval_vec,
        new_config_ptr->num_extra,
        new_config_ptr->expected_size,
        new_config_ptr->max_num_steps,
        new_config_ptr->max_ram_MB,
        new_config_ptr->pre_eval_func,
        new_config_ptr->capture_dense_output,
        new_config_ptr->force_retain_solver,
        new_config_ptr->events_vec,
        new_config_ptr->rtols,
        new_config_ptr->atols,
        new_config_ptr->max_step_size,
        new_config_ptr->first_step_size
    );
}

// ########################################################################################################################
// RKSolver (Base)
// ########################################################################################################################
/* ========================================================================= */
/* =========================  Protected Methods  =========================== */
/* ========================================================================= */
CyrkErrorCodes RKSolver::p_additional_setup() noexcept
{
    // Update stride information
    this->nstages_numy = this->n_stages * this->num_y;
    this->n_stages_p1  = this->n_stages + 1;

    // It is important to initialize the K variable with zeros
    std::fill(this->K_ptr, this->K_ptr + (this->num_y * this->n_stages_p1), 0.0);
    
    return CyrkErrorCodes::NO_ERROR;
}

void RKSolver::p_estimate_error() noexcept
{
    // Reset error norm
    this->error_norm = 0.0;

    // Initialize rtol and atol
    double rtol = this->rtols_ptr[0];
    double atol = this->atols_ptr[0];

    // Cache values thate used multiple times
    double* const y_old_ptr_ = this->y_old_ptr;
    double* const y_now_ptr_ = this->y_now_ptr;
    const double* const E_start_ptr = this->E_ptr;
    const double* const E_end_ptr   = this->E_ptr + this->n_stages_p1;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = this->use_array_rtols ? this->rtols_ptr[y_i] : rtol;
        atol = this->use_array_atols ? this->atols_ptr[y_i] : atol;

        // Dot product between K and E
        const size_t stride_K = y_i * this->n_stages_p1;
        double* const K_ptr_yi  = &this->K_ptr[stride_K];

        const double error_dot = std::inner_product(this->E_ptr, E_end_ptr, K_ptr_yi, 0.0);

        // Find scale of y for error calculations
        const double scale_inv = 1.0 / (atol + std::fmax(std::fabs(y_old_ptr_[y_i]), std::fabs(y_now_ptr_[y_i])) * rtol);

        // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
        // TODO: This will need to change if CySolver ever accepts complex numbers
        // error_norm_abs = fabs(error_dot_1)
        this->error_norm += ((error_dot * scale_inv) * (error_dot * scale_inv));
    }
    this->error_norm = this->step_size * std::sqrt(this->error_norm) / this->num_y_sqrt;
}


void RKSolver::p_calc_first_step_size() noexcept
{
    /*
        Select an initial step size based on the differential equation.
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.4.
    */

    if (this->num_y == 0) [[unlikely]]
    {
        this->step_size = INF;
    }
    else {
        double d0, d1, d2, d0_abs, d1_abs, d2_abs, scale;
        double h0, h0_direction, h1;
        double y_old_tmp;

        // Initialize tolerances to the 0 place. If `use_array_rtols` (or atols) is set then this will change in the loop.
        double rtol = this->rtols_ptr[0];
        double atol = this->atols_ptr[0];

        // Find the norm for d0 and d1
        d0 = 0.0;
        d1 = 0.0;
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            if (this->use_array_rtols)
            {
                rtol = this->rtols_ptr[y_i];
            }
            if (this->use_array_atols)
            {
                atol = this->atols_ptr[y_i];
            }

            y_old_tmp = this->y_old_ptr[y_i];
            scale = atol + std::fabs(y_old_tmp) * rtol;
            d0_abs = std::fabs(y_old_tmp / scale);
            d1_abs = std::fabs(this->dy_old_ptr[y_i] / scale);
            d0 += (d0_abs * d0_abs);
            d1 += (d1_abs * d1_abs);
        }

        d0 = std::sqrt(d0) / this->num_y_sqrt;
        d1 = std::sqrt(d1) / this->num_y_sqrt;

        if ((d0 < 1.0e-5) || (d1 < 1.0e-5))
        {
            h0 = 1.0e-6;
        }
        else {
            h0 = 0.01 * d0 / d1;
        }

        h0_direction = this->direction_flag ? h0 : -h0;

        this->t_now = this->t_old + h0_direction;
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            this->y_now_ptr[y_i] = this->y_old_ptr[y_i] + h0_direction * this->dy_old_ptr[y_i];
        }

        // Update dy
        this->diffeq(this);

        // Find the norm for d2
        d2 = 0.0;
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            if (this->use_array_rtols)
            {
                rtol = this->rtols_ptr[y_i];
            }
            if (this->use_array_atols)
            {
                atol = this->atols_ptr[y_i];
            }

            scale = atol + std::fabs(this->y_old_ptr[y_i]) * rtol;
            d2_abs = std::fabs((this->dy_now_ptr[y_i] - this->dy_old_ptr[y_i]) / scale);
            d2 += (d2_abs * d2_abs);
        }

        d2 = std::sqrt(d2) / (h0 * this->num_y_sqrt);

        if ((d1 <= 1.0e-15) && (d2 <= 1.0e-15))
        {
            h1 = std::fmax(1.0e-6, h0 * 1.0e-3);
        }
        else {
            h1 = std::pow((0.01 / std::fmax(d1, d2)), this->error_exponent);
        }
        this->step_size = std::fmax(10. * std::fabs(std::nextafter(this->t_old, this->direction_inf) - this->t_old), std::fmin(100.0 * h0, h1));
    }
}

void RKSolver::p_step_implementation() noexcept
{
    // Run RK integration step

    // Create local variables instead of calling class attributes for pointer objects.
    double* const l_K_ptr       = this->K_ptr;
    const double* const l_A_ptr = this->A_ptr;
    const double* const l_B_ptr = this->B_ptr;
    const double* const l_B_end_ptr = this->B_ptr + this->n_stages;
    const double* const l_C_ptr = this->C_ptr;
    double* const l_y_now_ptr   = this->y_now_ptr;
    double* const l_y_old_ptr   = this->y_old_ptr;
    double* const l_dy_now_ptr  = this->dy_now_ptr;
    double* const l_dy_old_ptr  = this->dy_old_ptr;

    // Determine step size based on previous loop
    // Find minimum step size based on the value of t (less floating point numbers between numbers when t is large)
    const double min_step_size = 10. * std::fabs(std::nextafter(this->t_old, this->direction_inf) - this->t_old);
    // Look for over/undershoots in previous step size
    this->step_size = std::clamp<double>(this->step_size, min_step_size, this->max_step_size);

    // Optimization variables
    // Define a very specific A (Row 1; Col 0) now since it is called consistently and does not change.
    const double A_at_10 = l_A_ptr[1 * this->len_Acols + 0];

    // Determine new step size
    bool step_accepted = false;
    bool step_rejected = false;
    bool step_error    = false;

    // !! Step Loop
    while (not step_accepted) {

        // Check if step size is too small
        // This will cause integration to fail: step size smaller than spacing between numbers
        if (this->step_size < min_step_size) [[unlikely]] {
            step_error = true;
            this->storage_ptr->update_status(CyrkErrorCodes::STEP_SIZE_ERROR_SPACING);
            break;
        }

        // Move time forward for this particular step size
        double t_delta_check;
        if (this->direction_flag) {
            this->step    = this->step_size;
            this->t_now   = this->t_old + this->step;
            t_delta_check = this->t_now - this->t_end;
        }
        else {
            this->step    = -this->step_size;
            this->t_now   = this->t_old + this->step;
            t_delta_check = this->t_end - this->t_now;
        }

        // Check that we are not at the end of integration with that move
        if (t_delta_check > 0.0) {
            this->t_now = this->t_end;

            // If we are, correct the step so that it just hits the end of integration.
            this->step = this->t_now - this->t_old;
            if (this->direction_flag) {
                this->step_size = this->step;
            }
            else {
                this->step_size = -this->step;
            }
        }

        // !! Calculate derivative using RK method

        // t_now must be updated for each loop of s in order to make the diffeq method calls.
        // But we need to return to its original value later on. Store in temp variable.
        const double time_tmp = this->t_now;

        for (size_t s = 1; s < this->len_C; s++) {
            // Find the current time based on the old time and the step size.
            this->t_now = this->t_old + l_C_ptr[s] * this->step;
            const size_t stride_A             = s * this->len_Acols;
            const double* const l_A_ptr_s     = &l_A_ptr[stride_A];
            const double* const l_A_ptr_s_end = l_A_ptr_s + s;

            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                double temp_double;
                const size_t stride_K             = y_i * this->n_stages_p1;
                const double* const l_K_ptr_yi     = &l_K_ptr[stride_K];
                const double* const l_A_ptr_s_end = l_A_ptr_s + s;

                // Dot Product (K, a) * step
                switch (s)
                {
                case 1:
                    // Set the first column of K
                    temp_double = l_dy_old_ptr[y_i];
                    l_K_ptr[stride_K] = temp_double;
                    temp_double *= A_at_10;
                    break;

                [[likely]] default:
                    // Dot product of A and K arrays up to s
                    temp_double = std::inner_product(l_A_ptr_s, l_A_ptr_s_end, l_K_ptr_yi, 0.0);
                    break;
                }

                // Update value of y_now
                l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
            }
            // Call diffeq method to update K with the new dydt
            // This will use the now updated dy_now_ptr based on the values of y_now_ptr and t_now_ptr.
            this->diffeq(this);

            // Update K based on the new dy values.
            double* K_col_ptr = l_K_ptr + s;
            double* dy_ptr = l_dy_now_ptr;
            size_t stride = this->n_stages_p1;

            for (size_t y_i = 0; y_i < this->num_y; ++y_i) {
                *K_col_ptr = *dy_ptr;
                K_col_ptr += stride;
                ++dy_ptr;
            }
        }

        // Restore t_now to its previous value.
        this->t_now = time_tmp;

        // Dot Product (K, B) * step
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            
            const size_t stride_K = y_i * this->n_stages_p1;
            const double* const l_K_ptr_yi = &l_K_ptr[stride_K];

            // Update y_now
            const double temp_double = std::inner_product(l_B_ptr, l_B_end_ptr, l_K_ptr_yi, 0.0);
            l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
        }

        // Find final dydt for this timestep
        // This will use the now updated dy_now_ptr based on the values of y_now_ptr and t_now_ptr.
        this->diffeq(this);

        // Set last column of K equal to dydt. K has size num_y * (n_stages + 1) so the last column is at n_stages
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t stride_K = y_i * this->n_stages_p1;
            l_K_ptr[stride_K + this->n_stages] = l_dy_now_ptr[y_i];
        }

        // Check how well this step performed by calculating its error.
        this->p_estimate_error();

        // Check the size of the error
        if (this->error_norm < 1.0) {
            // We found our step size because the error is low!
            // Update this step for the next time loop
            double step_factor = this->max_step_factor;
            if (this->error_norm == 0.0)
            {
                // Pass, leave as max.
            }
            else
            {
                // Estimate a new step size based on the error.
                const double error_safe = this->error_safety / std::pow(this->error_norm, this->error_exponent);
                step_factor = std::min<double>(this->max_step_factor, error_safe);
            }

            if (step_rejected) {
                // There were problems with this step size on the previous step loop. Make sure factor does
                //   not exasperate them.
                step_factor = std::min<double>(step_factor, 1.0);
            }

            // Update step size
            this->step_size *= step_factor;
            step_accepted = true;
        }
        else {
            // Error is still large. Keep searching for a better step size.
            const double error_safe = this->error_safety / std::pow(this->error_norm, this->error_exponent);
            this->step_size *= std::max<double>(this->min_step_factor, error_safe);
            step_rejected = true;
        }
    }

    // Update status depending if there were any errors.
    if (step_error) [[unlikely]]
    {
        // Issue with step convergence
        this->storage_ptr->update_status(CyrkErrorCodes::STEP_SIZE_ERROR_SPACING);
    }
    else if (!step_accepted) [[unlikely]]
    {
        // Issue with step convergence
        this->storage_ptr->update_status(CyrkErrorCodes::STEP_SIZE_ERROR_ACCEPTANCE);
    }

    // End of RK step.
}

/* ========================================================================= */
/* =========================  Public Methods  ============================== */
/* ========================================================================= */
/* Dense Output Methods */
void RKSolver::set_Q_order(size_t* Q_order_ptr)
{
    // Q's definition depends on the integrators implementation. 
    // For default RK, it is defined by Q = K.T.dot(self.P)  K has shape of (n_stages + 1, num_y) so K.T has shape of (num_y, n_stages + 1)
    // P has shape of (4, 3) for RK23; (7, 4) for RK45.. So (n_stages + 1, Q_order)
    // So Q has shape of (num_y, num_Pcols)
    Q_order_ptr[0] = this->len_Pcols;
}

void RKSolver::set_Q_array(double* Q_ptr)
{
    // Q's definition depends on the integrators implementation. 
    // For default RK, it is defined by Q = K.T.dot(self.P)  K has shape of (n_stages + 1, num_y) so K.T has shape of (num_y, n_stages + 1)
    // P has shape of (4, 3) for RK23; (7, 4) for RK45.. So (n_stages + 1, Q_order)
    // So Q has shape of (num_y, num_Pcols)

    switch (this->integration_method)
    {

    case(ODEMethod::RK23):
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t stride_Q = y_i * this->len_Pcols;
            const size_t stride_K = y_i * this->n_stages_p1;
            // len_Pcols == 3; n_stages + 1 == 4

            // P = 0
            double temp_double = this->K_ptr[stride_K] * this->P_ptr[0];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[3];
            Q_ptr[stride_Q] = temp_double;

            // P = 1
            size_t stride_P = this->n_stages_p1;
            temp_double  = this->K_ptr[stride_K]     * this->P_ptr[stride_P];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[stride_P + 1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[stride_P + 2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[stride_P + 3];
            Q_ptr[stride_Q + 1] = temp_double;

            // P = 2
            stride_P += this->n_stages_p1;
            temp_double  = this->K_ptr[stride_K]     * this->P_ptr[stride_P];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[stride_P + 1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[stride_P + 2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[stride_P + 3];
            Q_ptr[stride_Q + 2] = temp_double;
        }
        break;

    case(ODEMethod::RK45):
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t stride_Q = y_i * this->len_Pcols;
            const size_t stride_K = y_i * this->n_stages_p1;

            // len_Pcols == 4; n_stages + 1 == 7
            // P = 0
            double temp_double = this->K_ptr[stride_K] * this->P_ptr[0];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[3];
            temp_double += this->K_ptr[stride_K + 4] * this->P_ptr[4];
            temp_double += this->K_ptr[stride_K + 5] * this->P_ptr[5];
            temp_double += this->K_ptr[stride_K + 6] * this->P_ptr[6];
            Q_ptr[stride_Q] = temp_double;

            // P = 1
            size_t stride_P = this->n_stages_p1;
            temp_double  = this->K_ptr[stride_K]     * this->P_ptr[stride_P];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[stride_P + 1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[stride_P + 2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[stride_P + 3];
            temp_double += this->K_ptr[stride_K + 4] * this->P_ptr[stride_P + 4];
            temp_double += this->K_ptr[stride_K + 5] * this->P_ptr[stride_P + 5];
            temp_double += this->K_ptr[stride_K + 6] * this->P_ptr[stride_P + 6];
            Q_ptr[stride_Q + 1] = temp_double;

            // P = 2
            stride_P += this->n_stages_p1;
            temp_double  = this->K_ptr[stride_K]     * this->P_ptr[stride_P];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[stride_P + 1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[stride_P + 2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[stride_P + 3];
            temp_double += this->K_ptr[stride_K + 4] * this->P_ptr[stride_P + 4];
            temp_double += this->K_ptr[stride_K + 5] * this->P_ptr[stride_P + 5];
            temp_double += this->K_ptr[stride_K + 6] * this->P_ptr[stride_P + 6];
            Q_ptr[stride_Q + 2] = temp_double;

            // P = 3
            stride_P += this->n_stages_p1;
            temp_double  = this->K_ptr[stride_K]     * this->P_ptr[stride_P];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[stride_P + 1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[stride_P + 2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[stride_P + 3];
            temp_double += this->K_ptr[stride_K + 4] * this->P_ptr[stride_P + 4];
            temp_double += this->K_ptr[stride_K + 5] * this->P_ptr[stride_P + 5];
            temp_double += this->K_ptr[stride_K + 6] * this->P_ptr[stride_P + 6];
            Q_ptr[stride_Q + 3] = temp_double;
        }

        break;

    case(ODEMethod::DOP853):
        // We need this scope so we don't have to define things like `K_extended_ptr` for the other cases.
        {
            // This method uses the current values stored in K and expands upon them by 3 more values determined by calls to the diffeq.

            // We need to save a copy of the current state because we will overwrite the values shortly
            this->offload_to_temp();

            // We also need to store dy_dt so that they can be used in dot products
            double* K_extended_ptr        = &this->K[13 * this->num_y];
            double* temp_double_array_ptr = &this->K[16 * this->num_y];

            // S (row) == 13
            // Solve for dy used to call diffeq
            double temp_double;
            double temp_double_2;
            double temp_double_3;
            double temp_double_4;
            double K_ni;
            size_t stride_K;

            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                // Dot Product (K.T dot a) * h
                stride_K = this->n_stages_p1 * y_i;
                // Go up to a max of Row 13
                temp_double                        = 0.0;
                temp_double_array_ptr[y_i * 2]     = 0.0;
                temp_double_array_ptr[y_i * 2 + 1] = 0.0;
                for (size_t n_i = 0; n_i < this->n_stages_p1; n_i++)
                {
                    K_ni = K_ptr[stride_K + n_i];
                    temp_double                        += K_ni * this->AEXTRA_ptr[3 * n_i];
                    temp_double_array_ptr[y_i * 2]     += K_ni * this->AEXTRA_ptr[3 * n_i + 1];
                    temp_double_array_ptr[y_i * 2 + 1] += K_ni * this->AEXTRA_ptr[3 * n_i + 2];
                }

                // Update y for diffeq call using the temp_double dot product.
                this->y_now_ptr[y_i] = this->y_old_ptr[y_i] + temp_double * this->step;
            }
            // Update time and call the diffeq.
            this->t_now = this->t_old + (this->step * CEXTRA_ptr[0]);
            this->diffeq(this);

            // S (row) == 14
            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                // Store dy from the last call.
                K_ni = this->dy_now_ptr[y_i];
                K_extended_ptr[y_i * 3] = K_ni;

                // Dot Product (K.T dot a) * h
                // Add row 14 to the remaining dot product trackers
                temp_double_array_ptr[y_i * 2]     += K_ni * AEXTRA_ptr[3 * 13 + 1];
                temp_double_array_ptr[y_i * 2 + 1] += K_ni * AEXTRA_ptr[3 * 13 + 2];

                // Update y for diffeq call
                this->y_now_ptr[y_i] = this->y_old_ptr[y_i] + temp_double_array_ptr[y_i * 2] * this->step;
            }
            // Update time and call the diffeq.
            this->t_now = this->t_old + (this->step * CEXTRA_ptr[1]);
            this->diffeq(this);

            // S (row) == 15
            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                // Store dy from the last call.
                K_ni = this->dy_now_ptr[y_i];
                K_extended_ptr[y_i * 3 + 1] = K_ni;

                // Dot Product (K.T dot a) * h            
                // Add row 15 to the remaining dot product trackers
                temp_double_array_ptr[y_i * 2 + 1] += K_ni * AEXTRA_ptr[3 * 14 + 2];

                // Update y for diffeq call
                this->y_now_ptr[y_i] = this->y_old_ptr[y_i] + temp_double_array_ptr[y_i * 2 + 1] * this->step;
            }
            // Update time and call the diffeq.
            this->t_now = this->t_old + (this->step * CEXTRA_ptr[2]);
            this->diffeq(this);


            // Done with diffeq calls. Now build up Q matrix.
            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                // Store dy from the last call.
                K_extended_ptr[y_i * 3 + 2] = this->dy_now_ptr[y_i];

                // SciPy builds up a "F" matrix. Then in the dense interpolator this is reversed.
                // To keep consistency we will build F pre-reversed into Q. 
                // F (and Q) have the shape of (Interpolator Power, y_len); Interpolator power is 7
                const size_t stride_Q = y_i * this->len_Pcols;
                // F in normal direction is:
                // F[0] = delta_y
                // F[1] = h * f_old - delta_y
                // F[2] = 2 * delta_y - h * (self.f + f_old)
                // F[3:] = h * np.dot(self.D, K)
                // Reversed it would be
                // F[0:4] = reversed(h * np.dot(self.D, K))
                // F[4] = 2 * delta_y - h * (self.f + f_old)
                // F[5] = h * f_old - delta_y
                // F[6] = delta_y

                // Work on dot product between D and K
                stride_K = this->n_stages_p1 * y_i;
                // D Row 4
                temp_double   = 0.0;
                temp_double_2 = 0.0;
                temp_double_3 = 0.0;
                temp_double_4 = 0.0;

                // First add up normal K
                for (size_t n_i = 0; n_i < this->n_stages_p1; n_i++)
                {
                    K_ni = this->K_ptr[stride_K + n_i];
                    // Row 1
                    temp_double   += K_ni * this->D_ptr[4 * n_i];
                    // Row 2
                    temp_double_2 += K_ni * this->D_ptr[4 * n_i + 1];
                    // Row 3
                    temp_double_3 += K_ni * this->D_ptr[4 * n_i + 2];
                    // Row 4
                    temp_double_4 += K_ni * this->D_ptr[4 * n_i + 3];
                }
                // Now add the extra 3 rows from extended
                // Row 1
                temp_double   += K_extended_ptr[y_i * 3]     * this->D_ptr[4 * 13];
                temp_double_2 += K_extended_ptr[y_i * 3]     * this->D_ptr[4 * 13 + 1];
                temp_double_3 += K_extended_ptr[y_i * 3]     * this->D_ptr[4 * 13 + 2];
                temp_double_4 += K_extended_ptr[y_i * 3]     * this->D_ptr[4 * 13 + 3];
                // Row 2
                temp_double   += K_extended_ptr[y_i * 3 + 1] * this->D_ptr[4 * 14];
                temp_double_2 += K_extended_ptr[y_i * 3 + 1] * this->D_ptr[4 * 14 + 1];
                temp_double_3 += K_extended_ptr[y_i * 3 + 1] * this->D_ptr[4 * 14 + 2];
                temp_double_4 += K_extended_ptr[y_i * 3 + 1] * this->D_ptr[4 * 14 + 3];
                // Row 3
                temp_double   += K_extended_ptr[y_i * 3 + 2] * this->D_ptr[4 * 15];
                temp_double_2 += K_extended_ptr[y_i * 3 + 2] * this->D_ptr[4 * 15 + 1];
                temp_double_3 += K_extended_ptr[y_i * 3 + 2] * this->D_ptr[4 * 15 + 2];
                temp_double_4 += K_extended_ptr[y_i * 3 + 2] * this->D_ptr[4 * 15 + 3];


                // Store these in reversed order
                Q_ptr[stride_Q]     = this->step * temp_double_4;
                Q_ptr[stride_Q + 1] = this->step * temp_double_3;
                Q_ptr[stride_Q + 2] = this->step * temp_double_2;
                Q_ptr[stride_Q + 3] = this->step * temp_double;

                // Non dot product values
                // F[4] = 2 * delta_y - h * (self.f + f_old)
                // f_old = K[0]
                // delta_y requires the current y and last y, but the current y was just overwritten to find
                // K_extended. So we need to pull from the values we saved in temporary variables. Same thing with dy
                const double delta_y = this->y_tmp_ptr[y_i] - this->y_old_ptr[y_i];
                const double sum_dy  = this->dy_tmp_ptr[y_i] + this->K_ptr[stride_K];
                Q_ptr[stride_Q + 4]  = 2.0 * delta_y - this->step * sum_dy;

                // F[5] = h * f_old - delta_y
                Q_ptr[stride_Q + 5] = this->step * this->K_ptr[stride_K] - delta_y;

                // F[6] = delta_y
                Q_ptr[stride_Q + 6] = delta_y;
            }
        }

        // Return values that were saved in temp variables back to state variables.
        this->load_back_from_temp();
        break;

    [[unlikely]] default:
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            const size_t stride_K = this->n_stages_p1 * y_i;
            const size_t stride_Q = y_i * this->len_Pcols;

            for (size_t P_i = 0; P_i < this->len_Pcols; P_i++)
            {
                const size_t stride_P = P_i * this->n_stages_p1;
                // Initialize dot product
                double temp_double = 0.0;


                for (size_t n_i = 0; n_i < this->n_stages_p1; n_i++)
                {
                    temp_double += this->K_ptr[stride_K + n_i] * this->P_ptr[stride_P + n_i];
                }

                // Set equal to Q
                Q_ptr[stride_Q + P_i] = temp_double;
            }
        }
        break;
    }
}

CyrkErrorCodes RKSolver::setup()
{
    CyrkErrorCodes setup_status = CyrkErrorCodes::NO_ERROR;
    // Reset some parameters
    this->use_array_rtols = false;
    this->use_array_atols = false;

    // Call base class setup first
    setup_status = CySolverBase::setup();

    while (setup_status == CyrkErrorCodes::NO_ERROR)
    {
        // Reinterpret the config pointer for RKConfigs
        RKConfig* config_ptr = static_cast<RKConfig*>(this->storage_ptr->config_uptr.get());

        // Proceed with RK-specific setup tasks.
        // Check for errors
        this->user_provided_first_step_size = config_ptr->first_step_size;
        this->max_step_size = config_ptr->max_step_size;
        if (this->user_provided_first_step_size != 0.0) [[unlikely]]
        {
            if (this->user_provided_first_step_size < 0.0) [[unlikely]]
            {
                this->storage_ptr->update_status(CyrkErrorCodes::BAD_INITIAL_STEP_SIZE);
                break;
            }
            else if (this->user_provided_first_step_size > (this->t_delta_abs * 0.5)) [[unlikely]]
            {
                this->storage_ptr->update_status(CyrkErrorCodes::BAD_INITIAL_STEP_SIZE);
                break;
            }
        }

        // Setup tolerances
        // User can provide an array of relative tolerances, one for each y value.
        // The length of the pointer array must be the same as y0 (and <= 25).
        size_t num_rtols = config_ptr->rtols.size();
        size_t num_atols = config_ptr->atols.size();
        if ((num_rtols == 0) or 
            (num_rtols > this->num_y) or
            ((num_rtols > 1) and (num_rtols < this->num_y)) or
            (num_atols == 0) or
            (num_atols > this->num_y) or
            ((num_atols > 1) and (num_atols < this->num_y)))
        {
            // No rtols or atols provided, or the size of the array is not correct.
            setup_status = CyrkErrorCodes::BAD_CONFIG_DATA;
            break;
        }
        this->use_array_rtols = num_rtols > 1;
        this->use_array_atols = num_atols > 1;
        this->rtols_ptr = config_ptr->rtols.data();
        this->atols_ptr = config_ptr->atols.data();

        // Check for too small of rtols.
        for (size_t rtol_i = 0; rtol_i < num_rtols; rtol_i++)
        {
            double temp_double = this->rtols_ptr[rtol_i];
            if (temp_double < EPS_100) [[unlikely]]
            {
                temp_double = EPS_100;
            }
            this->rtols_ptr[rtol_i] = temp_double;
        }

        // Update initial step size
        if (this->user_provided_first_step_size == 0.0) [[likely]]
        {
            // User did not provide a step size. Try to find a good guess.
            this->p_calc_first_step_size();
        }
        else {
            this->step_size = this->user_provided_first_step_size;
        }
        break;
    }

    return setup_status;
}

// ########################################################################################################################
// Explicit Runge - Kutta 2(3)
// ########################################################################################################################
CyrkErrorCodes RK23::p_additional_setup() noexcept
{
    // Allocate the size of K
    try
    {
        this->K.resize(this->num_y * 4);
    }
    catch (const std::bad_alloc&)
    {
        // Memory allocation failed, return error code
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    // Setup RK constants before calling the base class reset
    this->order     = RK23_order;
    this->n_stages  = RK23_n_stages;
    this->len_Acols = RK23_len_Acols;
    this->len_Arows = RK23_len_Arows;
    this->len_C     = RK23_len_C;
    this->len_Pcols = RK23_len_Pcols;
    this->error_estimator_order = RK23_error_estimator_order;
    this->error_exponent        = RK23_error_exponent;

    this->C_ptr      = &this->RK23_C[0];
    this->A_ptr      = &this->RK23_A[0];
    this->B_ptr      = &this->RK23_B[0];
    this->E_ptr      = &this->RK23_E[0];
    this->E3_ptr     = nullptr;       // Not used for RK23
    this->E5_ptr     = nullptr;       // Not used for RK23
    this->P_ptr      = &this->RK23_P[0];
    this->D_ptr      = nullptr;       // Not used for RK23
    this->AEXTRA_ptr = nullptr;       // Not used for RK23
    this->CEXTRA_ptr = nullptr;       // Not used for RK23
    this->K_ptr      = this->K.data();
    this->integration_method = ODEMethod::RK23;

    return RKSolver::p_additional_setup();
}


// ########################################################################################################################
// Explicit Runge - Kutta 4(5)
// ########################################################################################################################
CyrkErrorCodes RK45::p_additional_setup() noexcept
{
    // Allocate the size of K
    try
    {
        this->K.resize(this->num_y * 7);
    }
    catch (const std::bad_alloc&)
    {
        // Memory allocation failed, return error code
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    // Setup RK constants before calling the base class reset
    this->order     = RK45_order;
    this->n_stages  = RK45_n_stages;
    this->len_Acols = RK45_len_Acols;
    this->len_Arows = RK45_len_Arows;
    this->len_C     = RK45_len_C;
    this->len_Pcols = RK45_len_Pcols;
    this->error_estimator_order = RK45_error_estimator_order;
    this->error_exponent        = RK45_error_exponent;

    this->C_ptr      = &this->RK45_C[0];
    this->A_ptr      = &this->RK45_A[0];
    this->B_ptr      = &this->RK45_B[0];
    this->E_ptr      = &this->RK45_E[0];
    this->E3_ptr     = nullptr;       // Not used for RK45
    this->E5_ptr     = nullptr;       // Not used for RK45
    this->P_ptr      = &this->RK45_P[0];
    this->D_ptr      = nullptr;       // Not used for RK45
    this->AEXTRA_ptr = nullptr;       // Not used for RK45
    this->CEXTRA_ptr = nullptr;       // Not used for RK45
    this->K_ptr      = this->K.data();
    this->integration_method = ODEMethod::RK45;

    return RKSolver::p_additional_setup();
}

// ########################################################################################################################
// Explicit Runge-Kutta Method of order 8(5,3) due Dormand & Prince
// ########################################################################################################################
CyrkErrorCodes DOP853::p_additional_setup() noexcept
{
    // Allocate the size of K
    try
    {
        this->K.resize(this->num_y * 18);
        // First 13 cols are K
        // Next 3 are K_extended
        // Next 2 are temp_double_array_ptr
    }
    catch (const std::bad_alloc&)
    {
        // Memory allocation failed, return error code
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    // Setup RK constants before calling the base class reset
    this->order     = DOP853_order;
    this->n_stages  = DOP853_n_stages;
    this->len_Acols = DOP853_len_Acols;
    this->len_Arows = DOP853_len_Arows;
    this->len_C     = DOP853_len_C;
    this->len_Pcols = DOP853_INTERPOLATOR_POWER; // Used by DOP853 dense output.
    this->error_estimator_order = DOP853_error_estimator_order;
    this->error_exponent        = DOP853_error_exponent;

    this->C_ptr      = &this->DOP853_C[0];
    this->A_ptr      = &this->DOP853_A[0];
    this->B_ptr      = &this->DOP853_B[0];
    this->E_ptr      = nullptr;        // Not used for DOP853
    this->E3_ptr     = &this->DOP853_E3[0];
    this->E5_ptr     = &this->DOP853_E5[0];
    this->P_ptr      = nullptr;        // TODO: Not implemented
    this->D_ptr      = &this->DOP853_D[0];
    this->AEXTRA_ptr = &this->DOP853_AEXTRA[0];
    this->CEXTRA_ptr = &this->DOP853_CEXTRA[0];
    this->K_ptr      = this->K.data();
    this->integration_method = ODEMethod::DOP853;

    return RKSolver::p_additional_setup();
}

void DOP853::p_estimate_error() noexcept
{
    double error_norm3 = 0.0;
    double error_norm5 = 0.0;

    // Initialize rtol and atol
    double rtol = this->rtols_ptr[0];
    double atol = this->atols_ptr[0];

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        if (this->use_array_rtols)
        {
            rtol = this->rtols_ptr[y_i];
        }

        if (this->use_array_atols)
        {
            atol = this->atols_ptr[y_i];
        }

        const size_t stride_K = y_i * this->n_stages_p1;
        double* const K_ptr_yi = &this->K_ptr[stride_K];

        // Dot product between K and E3 & E5 (sum over n_stages + 1; for DOP853 n_stages = 12
        // n = 0
        double error_dot3 = this->E3_ptr[0] * K_ptr_yi[0];
        double error_dot5 = this->E5_ptr[0] * K_ptr_yi[0];
        for (size_t n_i = 1; n_i < this->n_stages_p1; n_i++)
        {
            const double temp_double = K_ptr_yi[n_i];
            error_dot3 += this->E3_ptr[n_i] * temp_double;
            error_dot5 += this->E5_ptr[n_i] * temp_double;
        }

        // Find scale of y for error calculations
        const double scale_inv = 1.0 / (atol + std::fmax(std::fabs(this->y_old_ptr[y_i]), std::fabs(this->y_now_ptr[y_i])) * rtol);

        // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
        // TODO: This will need to change if CySolver ever accepts complex numbers
        // error_norm_abs = fabs(error_dot_1)
        error_dot3 *= scale_inv;
        error_dot5 *= scale_inv;

        error_norm3 += (error_dot3 * error_dot3);
        error_norm5 += (error_dot5 * error_dot5);
    }

    // Check if errors are zero
    if ((error_norm5 == 0.0) and (error_norm3) == 0.0)
    {
        this->error_norm = 0.0;
    }
    else
    {
        double error_denom = error_norm5 + 0.01 * error_norm3;
        this->error_norm = this->step_size * error_norm5 / std::sqrt(error_denom * this->num_y_dbl);
    }
}
