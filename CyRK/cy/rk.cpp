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
    double first_step_size_) :
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
    this->rtols = rtols_;
    this->atols = atols_;
    this->max_step_size = max_step_size_;
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

    // K_size may be different than n_stages_p1 (like DOP853)
    this->K_stride = this->K_size;

    // Allocate K and fill with zeros
    try {
        // Only resize if we need more space to avoid reallocation overhead
        size_t required_size = this->num_y * this->K_stride;
        this->K.resize(required_size);
        // It is important to initialize the K variable with zeros
        std::fill(this->K.data(), this->K.data() + required_size, 0.0);
    }
    catch (const std::bad_alloc&) {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }

    // Update pointer
    this->K_ptr = this->K.data();

    // Index pointer is used to find the pointer to strides of K quickly
    this->K_ptr_index.resize(this->num_y);
    this->K_ptr_index_ptr = this->K_ptr_index.data();
    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        this->K_ptr_index_ptr[y_i] = &this->K_ptr[y_i * this->K_stride];
    }

    // Set up other optimization variables
    if (this->A_ptr)
    {
        // Define a very specific A (Row 1; Col 0) now since it is called consistently and does not change.
        this->A_at_10 = this->A_ptr[1 * this->len_Acols + 0];
    }

    return CyrkErrorCodes::NO_ERROR;
}

void RKSolver::p_calc_first_step_size() noexcept
{
    /*
        Select an initial step size based on the differential equation.
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.4.
    */

    // Cache local vairables
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr       = this->dy_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    if (this->num_y == 0) [[unlikely]]
    {
        this->step_size = INF;
    }
    else {
        // Initialize tolerances to the 0 place. If `use_array_rtols` (or atols) is set then this will change in the loop.
        double rtol = l_rtols_ptr[0];
        double atol = l_atols_ptr[0];

        // Find the norm for d0 and d1
        double d0 = 0.0;
        double d1 = 0.0;
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
            atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

            const double y_old_tmp = l_y_old_ptr[y_i];
            const double scale = atol + std::abs(y_old_tmp) * rtol;

            // NOTE: We are removing the fabs because they are about to be squared anyways. But if we ever use complex numbers then we need to revisit this.
            // d0_abs = std::abs(y_old_tmp / scale);
            // d1_abs = std::abs(this->dy_old_ptr[y_i] / scale);
            const double d0_abs = y_old_tmp / scale;
            const double d1_abs = this->dy_old_ptr[y_i] / scale;
            d0 += (d0_abs * d0_abs);
            d1 += (d1_abs * d1_abs);
        }

        d0 = std::sqrt(d0) / this->num_y_sqrt;
        d1 = std::sqrt(d1) / this->num_y_sqrt;

        double h0 = 1.0e-6;
        if (not ((d0 < 1.0e-5) || (d1 < 1.0e-5)))
        {
            h0 = 0.01 * d0 / d1;
        }

        const double h0_direction = this->direction_flag ? h0 : -h0;

        this->t_now = this->t_old + h0_direction;
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + h0_direction * l_dy_old_ptr[y_i];
        }

        // Update dy
        this->call_diffeq();

        // Find the norm for d2
        double d2 = 0.0;
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

            const double scale = atol + std::abs(l_y_old_ptr[y_i]) * rtol;
            // NOTE: We are removing the fabs because they are about to be squared anyways. But if we ever use complex numbers then we need to revisit this.
            //d2_abs = std::abs((this->dy_now_ptr[y_i] - this->dy_old_ptr[y_i]) / scale);
            const double d2_abs = (l_dy_now_ptr[y_i] - l_dy_old_ptr[y_i]) / scale;
            d2 += (d2_abs * d2_abs);
        }

        d2 = std::sqrt(d2) / (h0 * this->num_y_sqrt);

        double h1;
        if ((d1 <= 1.0e-15) && (d2 <= 1.0e-15))
        {
            h1 = std::max(1.0e-6, h0 * 1.0e-3);
        }
        else {
            h1 = std::pow((0.01 / std::max(d1, d2)), this->error_exponent);
        }
        this->step_size = std::max(10. * std::abs(std::nextafter(this->t_old, this->direction_inf) - this->t_old), std::min(100.0 * h0, h1));
    }
}

void RKSolver::p_compute_stages() noexcept
{
    // Create local variables instead of calling class attributes for pointer objects.
    const double l_A_at_10                    = this->A_at_10;
    const size_t l_len_C                      = this->len_C;
    const size_t l_num_y                      = this->num_y;
    const size_t l_n_stages                   = this->n_stages;
    double* const CYRK_RESTRICT l_K_ptr       = this->K_ptr;
    const double* const CYRK_RESTRICT l_A_ptr = this->A_ptr;
    const double* const CYRK_RESTRICT l_B_ptr = this->B_ptr;
    const double* const CYRK_RESTRICT l_C_ptr = this->C_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr   = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr   = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr  = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr  = this->dy_old_ptr;
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;

    // t_now must be updated for each loop of s in order to make the diffeq method calls.
        // But we need to return to its original value later on. Store in temp variable.
    const double original_time = this->t_now;

    // !! Calculate derivative using RK Stages Method !!
    // Stage 1
    this->t_now = this->t_old + l_C_ptr[1] * this->step;
    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        const double temp_double = l_dy_old_ptr[y_i];
        // Set the first column of K (s=0)
        l_K_ptr_index_ptr[y_i][0] = temp_double;
        // Now update y
        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double * l_A_at_10);
    }
    // Call diffeq method to update K with the new dydt
        // This will use the now updated dy_now_ptr based on the values of y_now_ptr and t_now_ptr.
    this->call_diffeq();

    // Stage 2+
    for (size_t s = 2; s < l_len_C; s++)
    {
        // Find the current time based on the old time and the step size.
        this->t_now = this->t_old + l_C_ptr[s] * this->step;
        const size_t stride_A = s * this->len_Acols;
        const double* const l_A_ptr_s = &l_A_ptr[stride_A];

        for (size_t y_i = 0; y_i < l_num_y; y_i++)
        {
            double* const l_K_ptr_yi = l_K_ptr_index_ptr[y_i];

            // Update K based on the previous s loop value (including s=1 which is the loop above).
            l_K_ptr_yi[s - 1] = l_dy_now_ptr[y_i];

            // Dot Product (K, a) * step
            // Dot product of A and K arrays up to s
            double temp_double = l_A_ptr_s[0] * l_K_ptr_yi[0];
            for (size_t j = 1; j < s; j++)
            {
                temp_double += l_A_ptr_s[j] * l_K_ptr_yi[j];
            }

            // Update value of y_now
            l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
        }
        // Call diffeq method to update K with the new dydt
        // This will use the now updated dy_now_ptr based on the values of y_now_ptr and t_now_ptr.
        this->call_diffeq();
    }

    // Restore t_now to its previous value.
    this->t_now = original_time;

    // Dot Product (K, B) * step
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const l_K_ptr_yi = l_K_ptr_index_ptr[y_i];
        // Need to update K based on that last s step since it won't hit the loop update (where there is a [s - 1] index)
        l_K_ptr_yi[l_len_C - 1] = l_dy_now_ptr[y_i];

        // Update y_now
        double temp_double = l_B_ptr[0] * l_K_ptr_yi[0];
        for (size_t s = 1; s < l_n_stages; s++)
        {
            temp_double += l_B_ptr[s] * l_K_ptr_yi[s];
        }

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }

    // Find final dydt for this timestep
    // This will use the now updated dy_now_ptr based on the values of y_now_ptr and t_now_ptr.
    this->call_diffeq();

    // Set last column of K equal to dydt. K has size num_y * (n_stages + 1) so the last column is at n_stages
    for (size_t y_i = 0; y_i < l_num_y; y_i++) {
        l_K_ptr_index_ptr[y_i][l_n_stages] = l_dy_now_ptr[y_i];
    }
}

double RKSolver::p_estimate_error() noexcept
{
    // Cache values thate used multiple times
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    const double* const CYRK_RESTRICT l_E_ptr      = this->E_ptr;
    double* const CYRK_RESTRICT l_K_ptr            = this->K_ptr;
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    // Initialize rtol and atol
    double rtol = this->rtols_ptr[0];
    double atol = this->atols_ptr[0];

    // Inititalize error
    double l_error_norm = 0.0;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
        atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

        // Dot product between K and E
        const double* const l_K_ptr_yi = l_K_ptr_index_ptr[y_i];

        double error_dot = l_E_ptr[0] * l_K_ptr_yi[0];
        for (size_t s = 1; s < this->n_stages_p1; s++)
        {
            error_dot += l_E_ptr[s] * l_K_ptr_yi[s];
        }

        // Find scale of y for error calculations
        const double scale = error_dot / (atol + std::max(std::abs(l_y_old_ptr[y_i]), std::abs(l_y_now_ptr[y_i])) * rtol);

        // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
        // TODO: This will need to change if CySolver ever accepts complex numbers
        // error_norm_abs = fabs(error_dot_1)
        l_error_norm += (scale * scale);
    }
    return this->step_size * std::sqrt(l_error_norm) / this->num_y_sqrt;
}

void RKSolver::p_step_implementation() noexcept
{
    // Run RK integration step

    // Create local variables instead of calling class attributes for pointer objects.
    const double l_error_exponent             = this->error_exponent;
    const double l_max_step_factor            = this->max_step_factor;
    const double l_min_step_factor            = this->min_step_factor;

    // Determine step size based on previous loop
    // Find minimum step size based on the value of t (less floating point numbers between numbers when t is large)
    const double min_step_size = 10. * std::abs(std::nextafter(this->t_old, this->direction_inf) - this->t_old);
    // Look for over/undershoots in previous step size
    this->step_size = std::clamp<double>(this->step_size, min_step_size, this->max_step_size);

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
            this->step = this->step_size;
            this->t_now = this->t_old + this->step;
            t_delta_check = this->t_now - this->t_end;
        }
        else {
            this->step = -this->step_size;
            this->t_now = this->t_old + this->step;
            t_delta_check = this->t_end - this->t_now;
        }

        // Check that we are not at the end of integration with that move
        if (t_delta_check > 0.0) {
            this->t_now = this->t_end;

            // If we are, correct the step so that it just hits the end of integration.
            this->step = this->t_now - this->t_old;

            // Update the step size (absolute value of step).
            if (this->direction_flag) {
                this->step_size = this->step;
            }
            else {
                this->step_size = -this->step;
            }
        }

        // !! Calculate derivative using RK Stages Method !!
        this->p_compute_stages();

        // Check how well this step performed by calculating its error.
        const double error_norm = this->p_estimate_error();
        const double error_safe = this->error_safety / std::pow(error_norm, l_error_exponent);

        // Check the size of the error
        if (error_norm < 1.0)
        {
            // We found our step size because the error is low!
            // Update this step for the next time loop
            double step_factor = l_max_step_factor;
            // If error_norm == 0.0 then leave the step_factor as max_factor. Otherwise estimate a new one based on the error.
            if (error_norm != 0.0)
            {
                // Estimate a new step size based on the error.
                step_factor = std::min<double>(step_factor, error_safe);
            }

            if (step_rejected)
            {
                // There were problems with this step size on the previous step loop. Make sure factor does
                //   not exasperate them.
                step_factor = std::min<double>(step_factor, 1.0);
            }

            // Update step size
            this->step_size *= step_factor;
            step_accepted = true;
        }
        else
        {
            // Error is still large. Keep searching for a better step size.
            this->step_size *= std::max<double>(l_min_step_factor, error_safe);
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
                // Negative first step size. Even in reverse integration the step size should be positive.
                this->storage_ptr->update_status(CyrkErrorCodes::BAD_INITIAL_STEP_SIZE);
                break;
            }
            else if (this->user_provided_first_step_size > (this->t_delta_abs * 0.5)) [[unlikely]]
            {
                // First step size is greater than 50% of the solution domain.
                this->storage_ptr->update_status(CyrkErrorCodes::BAD_INITIAL_STEP_SIZE);
                break;
            }
        }

        // Setup tolerances
        // User can provide an array of relative tolerances, one for each y value.
        // The length of the pointer array must be the same as y0 (and <= 25).
        size_t num_rtols = config_ptr->rtols.size();
        size_t num_atols = config_ptr->atols.size();
        if (
            (num_rtols == 0) or
            ((num_rtols > 1) and (num_rtols != this->num_y)) or
            (num_atols == 0) or
            ((num_atols > 1) and (num_atols != this->num_y))
            )
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

/* Dense Output Methods */
void RKSolver::set_Q_order(size_t* Q_order_ptr)
{
    // Q's definition depends on the integrators implementation. 
    // For default RK, it is defined by Q = K.T.dot(self.P)  K has shape of (n_stages + 1, num_y) so K.T has shape of (num_y, n_stages + 1)
    // *Technically K is padded up to (K_stride, num_y) if n_stages + 1 is not divisible by 4.
    // P has shape of (4, 3) for RK23; (7, 4) for RK45.. So (n_stages + 1, Q_order)
    // So Q has shape of (num_y, num_Pcols)
    Q_order_ptr[0] = this->len_Pcols;
}

void RKSolver::set_Q_array(double* Q_ptr) noexcept
{
    // Create local cache of variables that will be used.
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    const double* const CYRK_RESTRICT l_P_ptr      = this->P_ptr;
    const size_t l_num_y       = this->num_y;
    const size_t l_n_stages_p1 = this->n_stages_p1;


    // Q's definition depends on the integrators implementation. 
    // For default RK, it is defined by Q = K.T.dot(self.P)  K has a (real) shape of (n_stages + 1, num_y) so K.T has shape of (num_y, n_stages + 1)
    // *Technically K is padded up to (K_stride, num_y) if n_stages + 1 is not divisible by 4.
    // P has shape of (4, 3) for RK23; (7, 4) for RK45.. So (n_stages + 1, Q_order)
    // So Q has shape of (num_y, num_Pcols)
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        const size_t stride_K = y_i * this->K_stride;
        const size_t stride_Q = y_i * this->len_Pcols;

        for (size_t P_i = 0; P_i < this->len_Pcols; P_i++)
        {
            const size_t stride_P = P_i * l_n_stages_p1;
            // Initialize dot product
            double temp_double = 0.0;


            for (size_t n_i = 0; n_i < l_n_stages_p1; n_i++)
            {
                temp_double += this->K_ptr[stride_K + n_i] * l_P_ptr[stride_P + n_i];
            }

            // Set equal to Q
            Q_ptr[stride_Q + P_i] = temp_double;
        }
    }
}


// ########################################################################################################################
// Explicit Runge - Kutta 2(3)
// ########################################################################################################################
CyrkErrorCodes RK23::p_additional_setup() noexcept
{
    // Setup RK constants before calling the base class reset
    this->order     = RK23_order;
    this->n_stages  = RK23_n_stages;
    this->len_Acols = RK23_len_Acols;
    this->len_Arows = RK23_len_Arows;
    this->len_C     = RK23_len_C;
    this->len_Pcols = RK23_len_Pcols;
    this->error_estimator_order = RK23_error_estimator_order;
    this->error_exponent        = RK23_error_exponent;

    this->K_size     = this->n_stages + 1;
    this->integration_method = ODEMethod::RK23;

    return RKSolver::p_additional_setup();
}

void RK23::p_compute_stages() noexcept
{
    // Create local pointers (omitting A, B, and C pointers since they are now hardcoded!)
    const size_t l_num_y                           = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr       = this->dy_old_ptr;
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;

    // t_now must be updated for each stage to make the diffeq method calls.
    // But we need to return to its original value later on. Store in temp variable.
    const double original_time = this->t_now;

    // ------------------------------------------------------------------------
    // Stage 1 (C[1] = 1/2)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (1.0 / 2.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // k_0 is the derivative from the start of the step
        k[0] = l_dy_old_ptr[y_i];

        // y_now = y_old + step * (A[1][0] * k_0)
        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * k[0] * (1.0 / 2.0));
    }
    // Calculate k_1
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 2 (C[2] = 3/4)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (3.0 / 4.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // Record k_1 from the previous diffeq call
        k[1] = l_dy_now_ptr[y_i];

        // y_now = y_old + step * (A[2][0] * k_0 + A[2][1] * k_1)
        // Since A[2][0] is 0.0, we completely skip k[0] here!
        const double temp_double = (3.0 / 4.0) * k[1];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    // Calculate k_2
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Final Update (B Vector Dot Product)
    // ------------------------------------------------------------------------
    // Restore t_now to its previous value for the final boundary evaluation.
    this->t_now = original_time;

    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // Record k_2 from the previous diffeq call
        k[2] = l_dy_now_ptr[y_i];

        // y_now = y_old + step * (B[0]*k_0 + B[1]*k_1 + B[2]*k_2)
        const double temp_double =
            (2.0 / 9.0) * k[0] +
            (1.0 / 3.0) * k[1] +
            (4.0 / 9.0) * k[2];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }

    // Find final dydt for this timestep (calculates k_3)
    this->call_diffeq();

    // Set last column of K equal to the final dydt. 
    // For RK23, n_stages is 3, so the last column is at index 3.
    for (size_t y_i = 0; y_i < l_num_y; y_i++) {
        l_K_ptr_index_ptr[y_i][3] = l_dy_now_ptr[y_i];
    }
}

double RK23::p_estimate_error() noexcept
{
    // Cache values that are used multiple times
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;

    // l_E_ptr is removed because we are hardcoding the E vector!
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    // Initialize rtol and atol
    double rtol = l_rtols_ptr[0];
    double atol = l_atols_ptr[0];

    // Initialize error
    double l_error_norm = 0.0;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
        atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

        const double* const l_K_ptr_yi = l_K_ptr_index_ptr[y_i];

        // --------------------------------------------------------------------
        // Unrolled Dot product between K and E
        // --------------------------------------------------------------------
        const double error_dot = 
            (5.0 / 72.0) * l_K_ptr_yi[0] +
            (-1.0 / 12.0) * l_K_ptr_yi[1] +
            (-1.0 / 9.0) * l_K_ptr_yi[2] +
            (1.0 / 8.0) * l_K_ptr_yi[3];

        // Find scale of y for error calculations
        const double scale = error_dot / (atol + std::max(std::abs(l_y_old_ptr[y_i]), std::abs(l_y_now_ptr[y_i])) * rtol);

        // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
        // TODO: This will need to change if CySolver ever accepts complex numbers
        l_error_norm += (scale * scale);
    }

    return this->step_size * std::sqrt(l_error_norm) / this->num_y_sqrt;
}

void RK23::set_Q_array(double* Q_ptr) noexcept
{
    // Create local cache of variables
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    const size_t l_num_y                           = this->num_y;

    // len_Pcols is strictly 3 for RK23.
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        const size_t stride_Q = y_i * 3;
        double* const k = l_K_ptr_index_ptr[y_i];

        // --------------------------------------------------------------------
        // Column 1 (P = 0)
        // P[0] is 1.0, and P[1], P[2], P[3] are 0.0. 
        // We skip the math completely
        // --------------------------------------------------------------------
        Q_ptr[stride_Q] = k[0];

        // --------------------------------------------------------------------
        // Column 2 (P = 1)
        // --------------------------------------------------------------------
        Q_ptr[stride_Q + 1] =
            (-4.0 / 3.0) * k[0] +
            k[1] +
            (4.0 / 3.0) * k[2] -
            k[3];

        // --------------------------------------------------------------------
        // Column 3 (P = 2)
        // --------------------------------------------------------------------
        Q_ptr[stride_Q + 2] =
            (5.0 / 9.0) * k[0] +
            (-2.0 / 3.0) * k[1] +
            (-8.0 / 9.0) * k[2] +
            k[3];
    }
}


// ########################################################################################################################
// Explicit Runge - Kutta 4(5)
// ########################################################################################################################
CyrkErrorCodes RK45::p_additional_setup() noexcept
{
    // Setup RK constants before calling the base class reset
    this->order     = RK45_order;
    this->n_stages  = RK45_n_stages;
    this->len_Acols = RK45_len_Acols;
    this->len_Arows = RK45_len_Arows;
    this->len_C     = RK45_len_C;
    this->len_Pcols = RK45_len_Pcols;
    this->error_estimator_order = RK45_error_estimator_order;
    this->error_exponent        = RK45_error_exponent;

    this->K_size     = this->n_stages + 1;
    this->integration_method = ODEMethod::RK45;

    return RKSolver::p_additional_setup();
}

void RK45::p_compute_stages() noexcept
{
    const size_t l_num_y                           = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr       = this->dy_old_ptr;
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;

    const double original_time = this->t_now;

    // ------------------------------------------------------------------------
    // Stage 1 (C[1] = 1/5)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (1.0 / 5.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // k_0
        k[0] = l_dy_old_ptr[y_i];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + this->step * ((1.0 / 5.0) * k[0]);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 2 (C[2] = 3/10)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (3.0 / 10.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // k_1
        k[1] = l_dy_now_ptr[y_i];

        const double temp_double =
            (3.0 / 40.0) * k[0] +
            (9.0 / 40.0) * k[1];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 3 (C[3] = 4/5)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (4.0 / 5.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // k_2
        k[2] = l_dy_now_ptr[y_i];

        const double temp_double =
            (44.0 / 45.0) * k[0] +
            (-56.0 / 15.0) * k[1] +
            (32.0 / 9.0) * k[2];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 4 (C[4] = 8/9)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (8.0 / 9.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // k_3
        k[3] = l_dy_now_ptr[y_i];

        const double temp_double =
            (19372.0 / 6561.0) * k[0] +
            (-25360.0 / 2187.0) * k[1] +
            (64448.0 / 6561.0) * k[2] +
            (-212.0 / 729.0) * k[3];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 5 (C[5] = 1.0)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (1.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // k_4
        k[4] = l_dy_now_ptr[y_i];

        const double temp_double =
            (9017.0 / 3168.0) * k[0] +
            (-355.0 / 33.0) * k[1] +
            (46732.0 / 5247.0) * k[2] +
            (49.0 / 176.0) * k[3] +
            (-5103.0 / 18656.0) * k[4];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Final Update (B Vector Dot Product)
    // ------------------------------------------------------------------------
    this->t_now = original_time;

    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // k_5
        k[5] = l_dy_now_ptr[y_i];

        // Completely skip k[1] because B[1] is 0.0
        const double temp_double =
            (35.0 / 384.0) * k[0] +
            (500.0 / 1113.0) * k[2] +
            (125.0 / 192.0) * k[3] +
            (-2187.0 / 6784.0) * k[4] +
            (11.0 / 84.0) * k[5];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }

    // Find final dydt for this timestep (calculates k_6)
    this->call_diffeq();

    // Set last column of K equal to the final dydt. 
    // For RK45, n_stages is 6, so the last column is at index 6.
    for (size_t y_i = 0; y_i < l_num_y; y_i++) {
        l_K_ptr_index_ptr[y_i][6] = l_dy_now_ptr[y_i];
    }
}

double RK45::p_estimate_error() noexcept
{
    // Cache values that are used multiple times
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;

    // l_E_ptr is removed because we are hardcoding the E vector!
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    // Initialize rtol and atol
    double rtol = l_rtols_ptr[0];
    double atol = l_atols_ptr[0];

    // Initialize error
    double l_error_norm = 0.0;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
        atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

        const double* const l_K_ptr_yi = l_K_ptr_index_ptr[y_i];

        // --------------------------------------------------------------------
        // Unrolled Dot product between K and E
        // Notice we skip l_K_ptr_yi[1] entirely because E[1] is 0.0
        // --------------------------------------------------------------------
        const double error_dot =
            (-71.0 / 57600.0) * l_K_ptr_yi[0] +
            (71.0 / 16695.0) * l_K_ptr_yi[2] +
            (-71.0 / 1920.0) * l_K_ptr_yi[3] +
            (17253.0 / 339200.0) * l_K_ptr_yi[4] +
            (-22.0 / 525.0) * l_K_ptr_yi[5] +
            (1.0 / 40.0) * l_K_ptr_yi[6];

        // Find scale of y for error calculations
        const double scale = error_dot / (atol + std::max(std::abs(l_y_old_ptr[y_i]), std::abs(l_y_now_ptr[y_i])) * rtol);

        // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
        // TODO: This will need to change if CySolver ever accepts complex numbers
        l_error_norm += (scale * scale);
    }

    return this->step_size * std::sqrt(l_error_norm) / this->num_y_sqrt;
}

void RK45::set_Q_array(double* Q_ptr) noexcept
{
    // Create local cache of variables
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    const size_t l_num_y                           = this->num_y;

    // len_Pcols is strictly 4 for RK45.
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        const size_t stride_Q = y_i * 4;
        double* const k = l_K_ptr_index_ptr[y_i];

        // --------------------------------------------------------------------
        // Column 1 (P = 0)
        // P[0] is 1.0, everything else is 0.0.
        // --------------------------------------------------------------------
        Q_ptr[stride_Q] = k[0];

        // --------------------------------------------------------------------
        // Column 2 (P = 1)
        // Notice we skip k[1] completely because P[1*7 + 1] is 0.0
        // --------------------------------------------------------------------
        Q_ptr[stride_Q + 1] =
            (-8048581381.0 / 2820520608.0) * k[0] +
            (131558114200.0 / 32700410799.0) * k[2] +
            (-1754552775.0 / 470086768.0) * k[3] +
            (127303824393.0 / 49829197408.0) * k[4] +
            (-282668133.0 / 205662961.0) * k[5] +
            (40617522.0 / 29380423.0) * k[6];

        // --------------------------------------------------------------------
        // Column 3 (P = 2)
        // Skipping k[1]
        // --------------------------------------------------------------------
        Q_ptr[stride_Q + 2] =
            (8663915743.0 / 2820520608.0) * k[0] +
            (-68118460800.0 / 10900136933.0) * k[2] +
            (14199869525.0 / 1410260304.0) * k[3] +
            (-318862633887.0 / 49829197408.0) * k[4] +
            (2019193451.0 / 616988883.0) * k[5] +
            (-110615467.0 / 29380423.0) * k[6];

        // --------------------------------------------------------------------
        // Column 4 (P = 3)
        // Skipping k[1]
        // --------------------------------------------------------------------
        Q_ptr[stride_Q + 3] =
            (-12715105075.0 / 11282082432.0) * k[0] +
            (87487479700.0 / 32700410799.0) * k[2] +
            (-10690763975.0 / 1880347072.0) * k[3] +
            (701980252875.0 / 199316789632.0) * k[4] +
            (-1453857185.0 / 822651844.0) * k[5] +
            (69997945.0 / 29380423.0) * k[6];
    }
}

// ########################################################################################################################
// Explicit Runge-Kutta Method of order 8(5,3) due Dormand & Prince
// ########################################################################################################################
CyrkErrorCodes DOP853::p_additional_setup() noexcept
{
    // Setup RK constants before calling the base class reset
    this->order     = DOP853_order;
    this->n_stages  = DOP853_n_stages;
    this->len_Acols = DOP853_len_Acols;
    this->len_Arows = DOP853_len_Arows;
    this->len_C     = DOP853_len_C;
    this->len_Pcols = DOP853_INTERPOLATOR_POWER; // Used by DOP853 dense output.
    this->error_estimator_order = DOP853_error_estimator_order;
    this->error_exponent        = DOP853_error_exponent;

    this->K_size     = (this->n_stages + 1) + 3 + 2; // First 13 cols are K; next 3 are K_extended; next 2 are temp_double_array_ptr
    this->integration_method = ODEMethod::DOP853;

    return RKSolver::p_additional_setup();
}

void DOP853::p_compute_stages() noexcept
{
    const size_t l_num_y                           = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr       = this->dy_old_ptr;
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;

    const double original_time = this->t_now;

    // ------------------------------------------------------------------------
    // Stage 1 (C[1])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.526001519587677318785587544488e-01) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[0] = l_dy_old_ptr[y_i]; // k_0

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + this->step * (5.26001519587677318785587544488e-2 * k[0]);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 2 (C[2])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.789002279381515978178381316732e-01) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[1] = l_dy_now_ptr[y_i]; // k_1

        const double temp_double =
            1.97250569845378994544595329183e-2 * k[0] +
            5.91751709536136983633785987549e-2 * k[1];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 3 (C[3])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.118350341907227396726757197510) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[2] = l_dy_now_ptr[y_i]; // k_2

        // Notice we skip k[1]
        const double temp_double =
            2.95875854768068491816892993775e-2 * k[0] +
            8.87627564304205475450678981324e-2 * k[2];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 4 (C[4])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.281649658092772603273242802490) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[3] = l_dy_now_ptr[y_i]; // k_3

        // Skipping k[1]
        const double temp_double = 
            2.41365134159266685502369798665e-1 * k[0] +
            -8.84549479328286085344864962717e-1 * k[2] +
            9.24834003261792003115737966543e-1 * k[3];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 5 (C[5])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.333333333333333333333333333333) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[4] = l_dy_now_ptr[y_i]; // k_4

        // Skipping k[1] and k[2]
        const double temp_double =
            3.7037037037037037037037037037e-2 * k[0] +
            1.70828608729473871279604482173e-1 * k[3] +
            1.25467687566822425016691814123e-1 * k[4];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 6 (C[6])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.25) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[5] = l_dy_now_ptr[y_i]; // k_5

        const double temp_double =
            3.7109375e-2 * k[0] +
            1.70252211019544039314978060272e-1 * k[3] +
            6.02165389804559606850219397283e-2 * k[4] +
            -1.7578125e-2 * k[5];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 7 (C[7])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.307692307692307692307692307692) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[6] = l_dy_now_ptr[y_i]; // k_6

        const double temp_double =
            3.70920001185047927108779319836e-2 * k[0] +
            1.70383925712239993810214054705e-1 * k[3] +
            1.07262030446373284651809199168e-1 * k[4] +
            -1.53194377486244017527936158236e-2 * k[5] +
            8.27378916381402288758473766002e-3 * k[6];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 8 (C[8])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.651282051282051282051282051282) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[7] = l_dy_now_ptr[y_i]; // k_7

        const double temp_double =
            6.24110958716075717114429577812e-1 * k[0] +
            -3.36089262944694129406857109825 * k[3] +
            -8.68219346841726006818189891453e-1 * k[4] +
            2.75920996994467083049415600797e1 * k[5] +
            2.01540675504778934086186788979e1 * k[6] +
            -4.34898841810699588477366255144e1 * k[7];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 9 (C[9])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.6) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[8] = l_dy_now_ptr[y_i]; // k_8

        const double temp_double =
            4.77662536438264365890433908527e-1 * k[0] +
            -2.48811461997166764192642586468 * k[3] +
            -5.90290826836842996371446475743e-1 * k[4] +
            2.12300514481811942347288949897e1 * k[5] +
            1.52792336328824235832596922938e1 * k[6] +
            -3.32882109689848629194453265587e1 * k[7] +
            -2.03312017085086261358222928593e-2 * k[8];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 10 (C[10])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (0.857142857142857142857142857142) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[9] = l_dy_now_ptr[y_i]; // k_9

        const double temp_double =
            -9.3714243008598732571704021658e-1 * k[0] +
            5.18637242884406370830023853209 * k[3] +
            1.09143734899672957818500254654 * k[4] +
            -8.14978701074692612513997267357 * k[5] +
            -1.85200656599969598641566180701e1 * k[6] +
            2.27394870993505042818970056734e1 * k[7] +
            2.49360555267965238987089396762 * k[8] +
            -3.0467644718982195003823669022 * k[9];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Stage 11 (C[11])
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (1.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[10] = l_dy_now_ptr[y_i]; // k_10

        const double temp_double =
            2.27331014751653820792359768449 * k[0] +
            -1.05344954667372501984066689879e1 * k[3] +
            -2.00087205822486249909675718444 * k[4] +
            -1.79589318631187989172765950534e1 * k[5] +
            2.79488845294199600508499808837e1 * k[6] +
            -2.85899827713502369474065508674 * k[7] +
            -8.87285693353062954433549289258 * k[8] +
            1.23605671757943030647266201528e1 * k[9] +
            6.43392746015763530355970484046e-1 * k[10];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Final Update (B Vector Dot Product)
    // ------------------------------------------------------------------------
    this->t_now = original_time;

    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[11] = l_dy_now_ptr[y_i]; // k_11

        // Notice we skip k[1], k[2], k[3], and k[4] because their B components are 0.0
        const double temp_double =
            5.42937341165687622380535766363e-2 * k[0] +
            4.45031289275240888144113950566 * k[5] +
            1.89151789931450038304281599044 * k[6] +
            -5.8012039600105847814672114227 * k[7] +
            3.1116436695781989440891606237e-1 * k[8] +
            -1.52160949662516078556178806805e-1 * k[9] +
            2.01365400804030348374776537501e-1 * k[10] +
            4.47106157277725905176885569043e-2 * k[11];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }

    // Find final dydt for this timestep (calculates k_12)
    this->call_diffeq();

    // Set last column of K equal to the final dydt. 
    // For DOP853, n_stages is 12, so the last column is at index 12.
    for (size_t y_i = 0; y_i < l_num_y; y_i++) {
        l_K_ptr_index_ptr[y_i][12] = l_dy_now_ptr[y_i];
    }
}

double DOP853::p_estimate_error() noexcept
{
    // Cache values that are used multiple times
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;

    // l_E3_ptr and l_E5_ptr are removed!
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    // Initialize rtol and atol
    double rtol = l_rtols_ptr[0];
    double atol = l_atols_ptr[0];

    // Initialize error
    double error_norm3 = 0.0;
    double error_norm5 = 0.0;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
        atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

        double* const k = l_K_ptr_index_ptr[y_i];

        // Find scale of y for error calculations
        const double scale_inv = 1.0 / (atol + std::max(std::abs(l_y_old_ptr[y_i]), std::abs(l_y_now_ptr[y_i])) * rtol);

        // --------------------------------------------------------------------
        // Unrolled Dot product for E3
        // Skips k[1], k[2], k[3], k[4], and k[12] entirely.
        // --------------------------------------------------------------------
        const double error_dot3 = scale_inv * (
            (5.42937341165687622380535766363e-2 - 0.244094488188976377952755905512) * k[0] +
            (4.45031289275240888144113950566) * k[5] +
            (1.89151789931450038304281599044) * k[6] +
            (-5.8012039600105847814672114227) * k[7] +
            (3.1116436695781989440891606237e-1 - 0.733846688281611857341361741547) * k[8] +
            (-1.52160949662516078556178806805e-1) * k[9] +
            (2.01365400804030348374776537501e-1) * k[10] +
            (4.47106157277725905176885569043e-2 - 0.220588235294117647058823529412e-1) * k[11]
            );

        // --------------------------------------------------------------------
        // Unrolled Dot product for E5
        // Skips the exact same indices as E3.
        // --------------------------------------------------------------------
        const double error_dot5 = scale_inv * (
            (0.1312004499419488073250102996e-1) * k[0] +
            (-0.1225156446376204440720569753e+1) * k[5] +
            (-0.4957589496572501915214079952) * k[6] +
            (0.1664377182454986536961530415e+1) * k[7] +
            (-0.3503288487499736816886487290) * k[8] +
            (0.3341791187130174790297318841) * k[9] +
            (0.8192320648511571246570742613e-1) * k[10] +
            (-0.2235530786388629525884427845e-1) * k[11]
            );
        
        error_norm3 += (error_dot3 * error_dot3);
        error_norm5 += (error_dot5 * error_dot5);
    }

    // Check if errors are zero
    if ((error_norm5 == 0.0) && (error_norm3 == 0.0))
    {
        return 0.0;
    }
    else
    {
        const double error_denom = error_norm5 + 0.01 * error_norm3;
        return this->step_size * error_norm5 / std::sqrt(error_denom * this->num_y_dbl);
    }
}

void DOP853::set_Q_array(double* Q_ptr) noexcept
{
    // We need to save a copy of the current state because we will overwrite the values shortly
    this->offload_to_temp();

    // Cache local variables
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    const size_t l_num_y                           = this->num_y;

    // ------------------------------------------------------------------------
    // Extra Stage 1 (Row 13 / S=13)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        // Unrolled Dot Product (K.T dot a) * h
        // Skips k[1], k[2], k[3], k[4] entirely!

        // Accumulator for y update 1 (skipping k[5] because AEXTRA[15] is 0.0)
        const double temp_double =
            k[0] * 5.61675022830479523392909219681e-2 +
            k[6] * 2.53500210216624811088794765333e-1 +
            k[7] * -2.46239037470802489917441475441e-1 +
            k[8] * -1.24191423263816360469010140626e-1 +
            k[9] * 1.5329179827876569731206322685e-1 +
            k[10] * 8.20105229563468988491666602057e-3 +
            k[11] * 7.56789766054569976138603589584e-3 +
            k[12] * -8.298e-3;

        // Accumulator for y update 2 (skipping k[8] and k[9] because they are 0.0)
        k[16] =
            k[0] * 3.18346481635021405060768473261e-2 +
            k[5] * 2.83009096723667755288322961402e-2 +
            k[6] * 5.35419883074385676223797384372e-2 +
            k[7] * -5.49237485713909884646569340306e-2 +
            k[10] * -1.08347328697249322858509316994e-4 +
            k[11] * 3.82571090835658412954920192323e-4 +
            k[12] * -3.40465008687404560802977114492e-4;

        // Accumulator for y update 3 (skipping k[9], k[10], k[11] because they are 0.0)
        k[17] =
            k[0] * -4.28896301583791923408573538692e-1 +
            k[5] * -4.69762141536116384314449447206 +
            k[6] * 7.68342119606259904184240953878 +
            k[7] * 4.06898981839711007970213554331 +
            k[8] * 3.56727187455281109270669543021e-1 +
            k[12] * -1.39902416515901462129418009734e-3;

        // Update y for diffeq call
        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + temp_double * this->step;
    }
    // CEXTRA[0] = 0.1
    this->t_now = this->t_old + (this->step * 0.1);
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Extra Stage 2 (Row 14 / S=14)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[13] = l_dy_now_ptr[y_i]; // Store dy

        // Add row 14 to the remaining dot product trackers
        k[16] += k[13] * 1.41312443674632500278074618366e-1;
        k[17] += k[13] * 2.9475147891527723389556272149;

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + k[16] * this->step;
    }
    // CEXTRA[1] = 0.2
    this->t_now = this->t_old + (this->step * 0.2);
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Extra Stage 3 (Row 15 / S=15)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[14] = l_dy_now_ptr[y_i]; // Store dy

        // Add row 15 to the remaining dot product tracker
        k[17] += k[14] * -9.15095847217987001081870187138;

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + k[17] * this->step;
    }
    // CEXTRA[2] = 0.777777777777777777777777777778
    this->t_now = this->t_old + (this->step * 0.777777777777777777777777777778);
    this->call_diffeq();

    // ------------------------------------------------------------------------
    // Build Dense Interpolator (Q Matrix)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[15] = l_dy_now_ptr[y_i]; // Final extra derivative

        // len_Pcols is 7 for DOP853 (interpolator power)
        const size_t stride_Q = y_i * 7;

        // Unrolled Dot Product between D and K
        // Notice we completely skip k[1], k[2], k[3], and k[4] for all 4 rows

        // D Row 1
        const double temp_double =
            k[0] * -0.84289382761090128651353491142e+1 +
            k[5] * 0.56671495351937776962531783590 +
            k[6] * -0.30689499459498916912797304727e+1 +
            k[7] * 0.23846676565120698287728149680e+1 +
            k[8] * 0.21170345824450282767155149946e+1 +
            k[9] * -0.87139158377797299206789907490 +
            k[10] * 0.22404374302607882758541771650e+1 +
            k[11] * 0.63157877876946881815570249290 +
            k[12] * -0.88990336451333310820698117400e-1 +
            k[13] * 0.18148505520854727256656404962e+2 +
            k[14] * -0.91946323924783554000451984436e+1 +
            k[15] * -0.44360363875948939664310572000e+1;

        // D Row 2
        const double temp_double_2 =
            k[0] * 0.10427508642579134603413151009e+2 +
            k[5] * 0.24228349177525818288430175319e+3 +
            k[6] * 0.16520045171727028198505394887e+3 +
            k[7] * -0.37454675472269020279518312152e+3 +
            k[8] * -0.22113666853125306036270938578e+2 +
            k[9] * 0.77334326684722638389603898808e+1 +
            k[10] * -0.30674084731089398182061213626e+2 +
            k[11] * -0.93321305264302278729567221706e+1 +
            k[12] * 0.15697238121770843886131091075e+2 +
            k[13] * -0.31139403219565177677282850411e+2 +
            k[14] * -0.93529243588444783865713862664e+1 +
            k[15] * 0.35816841486394083752465898540e+2;

        // D Row 3
        const double temp_double_3 =
            k[0] * 0.19985053242002433820987653617e+2 +
            k[5] * -0.38703730874935176555105901742e+3 +
            k[6] * -0.18917813819516756882830838328e+3 +
            k[7] * 0.52780815920542364900561016686e+3 +
            k[8] * -0.11573902539959630126141871134e+2 +
            k[9] * 0.68812326946963000169666922661e+1 +
            k[10] * -0.10006050966910838403183860980e+1 +
            k[11] * 0.77771377980534432092869265740 +
            k[12] * -0.27782057523535084065932004339e+1 +
            k[13] * -0.60196695231264120758267380846e+2 +
            k[14] * 0.84320405506677161018159903784e+2 +
            k[15] * 0.11992291136182789328035130030e+2;

        // D Row 4
        const double temp_double_4 =
            k[0] * -0.25693933462703749003312586129e+2 +
            k[5] * -0.15418974869023643374053993627e+3 +
            k[6] * -0.23152937917604549567536039109e+3 +
            k[7] * 0.35763911791061412378285349910e+3 +
            k[8] * 0.93405324183624310003907691704e+2 +
            k[9] * -0.37458323136451633156875139351e+2 +
            k[10] * 0.10409964950896230045147246184e+3 +
            k[11] * 0.29840293426660503123344363579e+2 +
            k[12] * -0.43533456590011143754432175058e+2 +
            k[13] * 0.96324553959188282948394950600e+2 +
            k[14] * -0.39177261675615439165231486172e+2 +
            k[15] * -0.14972683625798562581422125276e+3;

        // Store these in reversed order
        Q_ptr[stride_Q]     = this->step * temp_double_4;
        Q_ptr[stride_Q + 1] = this->step * temp_double_3;
        Q_ptr[stride_Q + 2] = this->step * temp_double_2;
        Q_ptr[stride_Q + 3] = this->step * temp_double;

        // Non dot product values
        const double delta_y = this->y_tmp_ptr[y_i] - l_y_old_ptr[y_i];
        const double sum_dy  = this->dy_tmp_ptr[y_i] + k[0];

        Q_ptr[stride_Q + 4] = 2.0 * delta_y - this->step * sum_dy;
        Q_ptr[stride_Q + 5] = this->step * k[0] - delta_y;
        Q_ptr[stride_Q + 6] = delta_y;
    }

    // Return values that were saved in temp variables back to state variables.
    this->load_back_from_temp();
}
