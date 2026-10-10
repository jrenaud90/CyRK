#include <stdexcept>
#include <numeric>
#include <algorithm>

#include "rk.hpp"
#include "dense.hpp"
#include "cysolution.hpp"

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

void RKSolver::p_compute_stages() noexcept
{
    // Create local variables instead of calling class attributes for pointer objects.
    const double l_A_at_10                    = this->A_at_10;
    const size_t l_len_C                      = this->len_C;
    const size_t l_num_y                      = this->num_y;
    const size_t l_n_stages                   = this->n_stages;
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
    this->diffeq(this);

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
        this->diffeq(this);
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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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
    this->diffeq(this);

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


// ########################################################################################################################
// Explicit Runge-Kutta Method of order 5(4) due to Tsitouras
// ########################################################################################################################
/* Tableau and error weights: the 80-digit coefficients of OrdinaryDiffEq.jl (MIT license, "tsit_tableaus.jl"), which
   satisfy the order conditions to 1e-76. The interpolant weights b_i(theta) solve the order 1-4 conditions together
   with b_i'(0) = delta_i1, b_i'(1) = delta_i7, and a theta^4 coefficient of 0.1017 in b_2 (Tsitouras 2011); they match
   OrdinaryDiffEq.jl's to double precision. */
CyrkErrorCodes Tsit5::p_additional_setup() noexcept
{
    this->order     = Tsit5_order;
    this->n_stages  = Tsit5_n_stages;
    this->len_Pcols = Tsit5_len_Pcols;
    this->error_estimator_order = Tsit5_error_estimator_order;
    this->error_exponent        = Tsit5_error_exponent;

    this->K_size     = this->n_stages + 1;
    this->integration_method = ODEMethod::TSIT5;

    return RKSolver::p_additional_setup();
}

void Tsit5::p_compute_stages() noexcept
{
    const size_t l_num_y                           = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr       = this->dy_old_ptr;
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;

    const double original_time = this->t_now;

    // ------------------------------------------------------------------------
    // Stage 1 (C = 161/1000)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (161.0 / 1000.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[0] = l_dy_old_ptr[y_i];

        const double temp_double =
            (161.0 / 1000.0) * k[0];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 2 (C = 327/1000)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (327.0 / 1000.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[1] = l_dy_now_ptr[y_i];

        const double temp_double =
            -8.48065549235698854442687425023077468e-3 * k[0] +
            3.35480655492356988544426874250230775e-1 * k[1];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 3 (C = 9/10)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (9.0 / 10.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[2] = l_dy_now_ptr[y_i];

        const double temp_double =
            2.89715305710549343213043259419293876 * k[0] +
            -6.35944848997507484314815991238382563 * k[1] +
            4.36229543286958141101772731819088686 * k[2];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 4 (C = 0.980026)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 9.80025540904509685729810286287024595e-1 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[3] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.32586482843925660442887792084051132 * k[0] +
            -1.1748883564062827877747170339785773e1 * k[1] +
            7.49553934288983620830460478456435816 * k[2] +
            -9.24950663617552492565020793320719161e-2 * k[3];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 5 (C = 1)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 1.0 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[4] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.86145544294642002865925148698264789 * k[0] +
            -1.29209693178471092917061186817833594e1 * k[1] +
            8.15936789857615864318040079453925349 * k[2] +
            -7.15849732814009972245305425258297387e-2 * k[3] +
            -2.82690503940683829090030572127122415e-2 * k[4];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Final Update (B Vector Dot Product)
    // ------------------------------------------------------------------------
    this->t_now = original_time;

    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[5] = l_dy_now_ptr[y_i];

        const double temp_double =
            9.64607668180652295181673131651287633e-2 * k[0] +
            (1.0 / 100.0) * k[1] +
            4.7988965041449957477524953229059652e-1 * k[2] +
            1.37900857410374189319227482185687277 * k[3] +
            -3.29006951543608067990104758571136385 * k[4] +
            2.3247105240997739824153559183987658 * k[5];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }

    // Derivative at the end of the step (k_6); it is also the first stage of the next step.
    this->diffeq(this);

    for (size_t y_i = 0; y_i < l_num_y; y_i++) {
        l_K_ptr_index_ptr[y_i][6] = l_dy_now_ptr[y_i];
    }
}

double Tsit5::p_estimate_error() noexcept
{
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    double rtol = l_rtols_ptr[0];
    double atol = l_atols_ptr[0];
    double l_error_norm = 0.0;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
        atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

        const double* const k = l_K_ptr_index_ptr[y_i];

        // Difference between the propagated and embedded solutions (b - b_hat) dotted with K
        const double error_dot =
            -1.78001105222577144337855060753953478e-3 * k[0] +
            -8.1643445965674690322363606335468624e-4 * k[1] +
            7.88087801026199601031472767252630424e-3 * k[2] +
            -1.44711007173262907537165147972635117e-1 * k[3] +
            5.82357165452555225019937610652042179e-1 * k[4] +
            -4.58082105929186946661636518832554297e-1 * k[5] +
            (1.0 / 66.0) * k[6];

        const double scale = error_dot / (
            atol + std::max(std::abs(l_y_old_ptr[y_i]), std::abs(l_y_now_ptr[y_i])) * rtol);
        l_error_norm += (scale * scale);
    }

    return this->step_size * std::sqrt(l_error_norm) / this->num_y_sqrt;
}

void Tsit5::set_Q_array(double* Q_ptr) noexcept
{
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    const size_t l_num_y                           = this->num_y;

    // Q[j] holds sum_i b_i(theta_j) k_i at the interpolation nodes theta_j ("rk.hpp", Tsit5_dense_nodes).
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        const size_t stride_Q = y_i * 4;
        double* const k = l_K_ptr_index_ptr[y_i];

        Q_ptr[stride_Q + 0] =
            9.58537827136971256979571589976441736e-2 * k[0] +
            (2386608057.0 / 1099511627776.0) * k[1] +
            6.68078327239371566247749120119790133e-2 * k[2] +
            -1.78572685706789428336955588672067894e-1 * k[3] +
            5.49512441230556638311989129460933574e-1 * k[4] +
            -4.10052335997784914855030178219997656e-1 * k[5] +
            2.07647326533333398401737213134765625e-2 * k[6];
        Q_ptr[stride_Q + 1] =
            1.07412352300968767432365512674872597e-1 * k[0] +
            (1817.0 / 160000.0) * k[1] +
            3.95609030560453086955346333182060506e-1 * k[2] +
            -3.44752143525935473298068705464218193e-1 * k[3] +
            1.31618536495816473621707507390674513 * k[4] +
            -1.01706085429365111730671821429946004 * k[5] +
            3.125e-2 * k[6];
        Q_ptr[stride_Q + 2] =
            9.27106096810543914271529897180151905e-2 * k[0] +
            1.10088756034747348167002201080322266e-2 * k[1] +
            4.90947177380853913281667692411788601e-1 * k[2] +
            1.04023207991263215606868454093458821 * k[3] +
            -2.35833924289610798589962609989133261 * k[4] +
            1.64458917868599205544571568540543182 * k[5] +
            -6.76330533678992651402950286865234375e-2 * k[6];
        Q_ptr[stride_Q + 3] =
            9.64607668180652295181673131651287633e-2 * k[0] +
            (1.0 / 100.0) * k[1] +
            4.7988965041449957477524953229059652e-1 * k[2] +
            1.37900857410374189319227482185687277 * k[3] +
            -3.29006951543608067990104758571136385 * k[4] +
            2.3247105240997739824153559183987658 * k[5];
    }
}


// ########################################################################################################################
// Explicit Runge-Kutta Method of order 7(6) due to Verner
// ########################################################################################################################
/* Tableau and error weights: Verner's exact rationals ("most efficient" pair RKV76.IIa). Extra stages and interpolant:
   Verner's 40-digit coefficients as corrected in 2024 (file RKV76.IIa.Efficient.00001675585.240711). K holds the ten
   stages, then the derivative at the end of the step (Verner's stage 11), then the five extra stages. */
CyrkErrorCodes Vern7::p_additional_setup() noexcept
{
    this->order     = Vern7_order;
    this->n_stages  = Vern7_n_stages;
    this->len_Pcols = Vern7_len_Pcols;
    this->error_estimator_order = Vern7_error_estimator_order;
    this->error_exponent        = Vern7_error_exponent;

    this->K_size     = this->n_stages + 1 + Vern7_nEXTRA_stages;
    this->integration_method = ODEMethod::VERN7;

    return RKSolver::p_additional_setup();
}

void Vern7::p_compute_stages() noexcept
{
    const size_t l_num_y                           = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr       = this->dy_old_ptr;
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;

    const double original_time = this->t_now;

    // ------------------------------------------------------------------------
    // Stage 1 (C = 1/200)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (1.0 / 200.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[0] = l_dy_old_ptr[y_i];

        const double temp_double =
            (1.0 / 200.0) * k[0];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 2 (C = 0.108889)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 1.08888888888888888888888888888888889e-1 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[1] = l_dy_now_ptr[y_i];

        const double temp_double =
            (-4361.0 / 4050.0) * k[0] +
            (2401.0 / 2025.0) * k[1];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 3 (C = 0.163333)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 1.63333333333333333333333333333333333e-1 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[2] = l_dy_now_ptr[y_i];

        const double temp_double =
            (49.0 / 1200.0) * k[0] +
            (49.0 / 400.0) * k[2];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 4 (C = 911/2000)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (911.0 / 2000.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[3] = l_dy_now_ptr[y_i];

        const double temp_double =
            (2454451729.0 / 3841600000.0) * k[0] +
            (-9433712007.0 / 3841600000.0) * k[2] +
            (4364554539.0 / 1920800000.0) * k[3];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 5 (C = 0.609509)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 6.09509448997838131708700442148602495e-1 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[4] = l_dy_now_ptr[y_i];

        const double temp_double =
            -2.66157737501875713111925929786181812 * k[0] +
            1.08045138864561376956539665536553284e1 * k[2] +
            -8.35391465739619941196804854781929169 * k[3] +
            8.20487594956656979142041734174383921e-1 * k[4];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 6 (C = 221/250)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (221.0 / 250.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[5] = l_dy_now_ptr[y_i];

        const double temp_double =
            6.06774143469677099271836018387727671 * k[0] +
            -2.471127363591108579734203485290746e1 * k[2] +
            2.04275179307888939404577311174834661e1 * k[3] +
            -1.90615797881664715062409678435275701 * k[4] +
            1.00617224924206801479004033589947419 * k[5];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 7 (C = 37/40)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (37.0 / 40.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[6] = l_dy_now_ptr[y_i];

        const double temp_double =
            1.20546700762532029950910945289277831e1 * k[0] +
            -4.97547849504689893280725761533144476e1 * k[2] +
            4.11428886386046766325969841671015735e1 * k[3] +
            -4.46176014997400418564191160348481538 * k[4] +
            2.04233482223917495982171707770860854 * k[5] +
            -9.8348436654061073795308016938702244e-2 * k[6];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 8 (C = 1)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 1.0 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[7] = l_dy_now_ptr[y_i];

        const double temp_double =
            1.01381465228818078764184514198168903e1 * k[0] +
            -4.26411360317175021462284600673663573e1 * k[2] +
            3.57638400399225700713502117802316005e1 * k[3] +
            -4.34802284039290765334037029690824594 * k[4] +
            2.00986226837703589544194359301182755 * k[5] +
            3.48749046033827240595382285305314588e-1 * k[6] +
            -2.71439005104831284237158714091029741e-1 * k[7];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 9 (C = 1)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 1.0 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[8] = l_dy_now_ptr[y_i];

        const double temp_double =
            -4.50300720342986771243532240507376964e1 * k[0] +
            1.873272437654588840752418206154202e2 * k[2] +
            -1.5402882369350186905967286210345104e2 * k[3] +
            1.85646530634753623385949233295843914e1 * k[4] +
            -7.14180967929507885492542049682355119 * k[5] +
            1.3088085781613786251147627060076967 * k[6];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Final Update (B Vector Dot Product)
    // ------------------------------------------------------------------------
    this->t_now = original_time;

    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[9] = l_dy_now_ptr[y_i];

        const double temp_double =
            4.71556184862722217043176510883817568e-2 * k[0] +
            2.57505642984341518959643610103768758e-1 * k[3] +
            2.62166539774126204771386309576452771e-1 * k[4] +
            1.52160926567385574032313319916511754e-1 * k[5] +
            4.93996917003248424690717589322787684e-1 * k[6] +
            -2.94303117140325044155724474409270343e-1 * k[7] +
            8.13174723249510999973459944013676189e-2 * k[8];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }

    // Derivative at the end of the step (k_10); it is also the first stage of the next step.
    this->diffeq(this);

    for (size_t y_i = 0; y_i < l_num_y; y_i++) {
        l_K_ptr_index_ptr[y_i][10] = l_dy_now_ptr[y_i];
    }
}

double Vern7::p_estimate_error() noexcept
{
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    double rtol = l_rtols_ptr[0];
    double atol = l_atols_ptr[0];
    double l_error_norm = 0.0;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
        atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

        const double* const k = l_K_ptr_index_ptr[y_i];

        // Difference between the propagated and embedded solutions (b - b_hat) dotted with K
        const double error_dot =
            2.54701187993104541699947511358977898e-3 * k[0] +
            -9.65839487279574909126661599061503188e-3 * k[3] +
            4.20647097563969027734147319113774615e-2 * k[4] +
            -6.66822437469301090659987634347776289e-2 * k[5] +
            2.65009746462128136352900200346432448e-1 * k[6] +
            -2.94303117140325044155724474409270343e-1 * k[7] +
            8.13174723249510999973459944013676189e-2 * k[8] +
            -2.02951846633562822276705479381043036e-2 * k[9];

        const double scale = error_dot / (
            atol + std::max(std::abs(l_y_old_ptr[y_i]), std::abs(l_y_now_ptr[y_i])) * rtol);
        l_error_norm += (scale * scale);
    }

    return this->step_size * std::sqrt(l_error_norm) / this->num_y_sqrt;
}

void Vern7::set_Q_array(double* Q_ptr) noexcept
{
    // The extra stages overwrite the solver's current state, so hold a copy until they are done.
    this->offload_to_temp();

    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    const size_t l_num_y                           = this->num_y;

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 12, stored in K[11] (C = 0.671663)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        const double temp_double =
            4.60882848003009778457381016068200459e-2 * k[0] +
            2.6095211559122214118077140161591946e-1 * k[3] +
            2.59353988657214895410432536477717266e-1 * k[4] +
            1.07128485323135859014952677616400962e-1 * k[5] +
            -7.33573700780350194001014782374152843e-2 * k[6] +
            9.51681759723114884817765750556303199e-2 * k[7] +
            -5.55747092373399776404042587410045485e-2 * k[8] +
            3.19036654750643418777009120669178258e-2 * k[10];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + 6.71662636503874706770866467460986047e-1 * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 13, stored in K[12] (C = 1/8)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[11] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.83733397559838718332258000949993407e-2 * k[0] +
            8.57752423669400049924405659909930229e-2 * k[3] +
            -3.42920458100212298795819832282465864e-2 * k[4] +
            -2.8182597311030361514853545674702714e-2 * k[5] +
            -2.0781161488023721821264109765760484e-2 * k[6] +
            -3.67074014444101961902679584597478886e-3 * k[7] +
            1.14064525843646881910238233995234162e-2 * k[8] +
            -2.75015391337440809870189066399182639e-3 * k[10] +
            5.91216639596021759167381356931606199e-2 * k[11];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (1.0 / 8.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 14, stored in K[13] (C = 1/4)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[12] = l_dy_now_ptr[y_i];

        const double temp_double =
            3.78162016789448438909773421753747594e-2 * k[0] +
            3.13467853263107410377564263687262256e-2 * k[3] +
            2.22511007752623818722616161356840683e-2 * k[4] +
            9.9581319554210005293012046228797182e-3 * k[5] +
            1.52230960438999255828742600414309319e-2 * k[6] +
            -7.54704745411831712587319413260945673e-3 * k[7] +
            1.31588820654662970543159639748314856e-3 * k[8] +
            -2.34751903335989691933291152271596881e-3 * k[10] +
            -2.4269839327928092842547847933424482e-2 * k[11] +
            1.66253201829020784269151507847171056e-1 * k[12];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (1.0 / 4.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 15, stored in K[14] (C = 53/100)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[13] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.34507508552560728197726679363380342e-2 * k[0] +
            3.4301733821384506727609198836100659e-1 * k[3] +
            2.07674272620339857430078773211759434e-1 * k[4] +
            7.72267561939583633488638687371532844e-2 * k[5] +
            1.32726879856113532176413536463869672e-4 * k[6] +
            2.22199582024624012910229048629165017e-2 * k[7] +
            -1.74102288684023244341141420229254057e-2 * k[8] +
            7.19759376991503078555060294567268242e-3 * k[10] +
            -8.64550486846266576128018281473015066e-2 * k[11] +
            -7.70541191826039244366412494210834848e-2 * k[12];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (53.0 / 100.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 16, stored in K[15] (C = 79/100)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[14] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.68108791163588423500918758538759352e-2 * k[0] +
            3.68522190161843244378422161500029585e-1 * k[3] +
            2.29406053010307533100880807056839154e-1 * k[4] +
            8.8544250143138025319602117114776354e-2 * k[5] +
            2.93787928309864416585534485484152205e-2 * k[6] +
            5.46345674500464761660272388700065283e-3 * k[7] +
            -1.31175004963559499554911464731445366e-2 * k[8] +
            5.10909368974529678430762443843873064e-3 * k[10] +
            1.24196408958775325673055283581699496e-1 * k[11] +
            -1.04313624159803406926024895507930591e-1 * k[12];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (79.0 / 100.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Build Dense Interpolator (Q Matrix): Q[j] = sum_i b_i(theta_j) k_i at the nodes Vern7_dense_nodes
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[15] = l_dy_now_ptr[y_i];

        const size_t stride_Q = y_i * 7;

        Q_ptr[stride_Q + 0] =
            3.27757669842538634747854411222140259e-2 * k[0] +
            2.25923127679703324018165376009379681e-2 * k[3] +
            2.30012375466030384529049841296213701e-2 * k[4] +
            1.33498714988687839245404339827294134e-2 * k[5] +
            4.33409253716009751241885558880053156e-2 * k[6] +
            -2.58207470483555482868800345627975119e-2 * k[7] +
            7.1344058599049064188584857761414268e-3 * k[8] +
            -5.2974445795571314300708965715144478e-3 * k[10] +
            1.15139675958332918802284258064042885e-2 * k[11] +
            1.6503608823377248170171772050320992e-2 * k[12] +
            -3.00080493497891683934622462948624546e-2 * k[13] +
            -2.77222037795159516565725595166313092e-2 * k[14] +
            -3.15589641911946400805088994105690769e-2 * k[15];
        Q_ptr[stride_Q + 1] =
            4.34630647758150122408676206332032917e-2 * k[0] +
            9.09318818126924661736131800243891837e-2 * k[3] +
            9.2577764641192775184671585537403495e-2 * k[4] +
            5.37319463402075535461692057792255506e-2 * k[5] +
            1.74443048129649524964800697128516527e-1 * k[6] +
            -1.03926018687436480299026144471911449e-1 * k[7] +
            2.87152960885171040772767807040661162e-2 * k[8] +
            -2.06422142999256376289353358333703265e-2 * k[10] +
            7.21691070300476536231684342305232081e-2 * k[11] +
            1.03443986761554926174770621625964377e-1 * k[12] +
            -7.03475911068230388668999953021022246e-2 * k[13] +
            -1.38962565078841338541207956955887744e-1 * k[14] +
            -1.37121143906650520649268693100020004e-1 * k[15];
        Q_ptr[stride_Q + 2] =
            4.71580073896748559967608656024694591e-2 * k[0] +
            1.09986929520816191445898653162523179e-1 * k[3] +
            1.1197771201700024325874959861266878e-1 * k[4] +
            6.49916363471956822862534369796752613e-2 * k[5] +
            2.10998110426809870209860091661368632e-1 * k[6] +
            -1.25704026628409756386640455893412673e-1 * k[7] +
            3.47326722387950880280335341489934985e-2 * k[8] +
            -2.56354480391724614394701295334245293e-2 * k[10] +
            5.18197399013559327921735152490177752e-2 * k[11] +
            7.42761093900089394249524773142155925e-2 * k[12] +
            1.01706953267316160737207020764019755e-1 * k[13] +
            -1.13547708686803936722871723113281444e-1 * k[14] +
            -1.54088812144586809630906884954833286e-1 * k[15];
        Q_ptr[stride_Q + 3] =
            5.01107346493994414140939385937789851e-2 * k[0] +
            1.22300907491465844858959842018667911e-1 * k[3] +
            1.24514575124176284764676994760641186e-1 * k[4] +
            7.22680062008006475327977987432466544e-2 * k[5] +
            2.34621154500903218490391791307877039e-1 * k[6] +
            -1.39777668118976465502783744917991894e-1 * k[7] +
            3.86212921200271687588446423801743549e-2 * k[8] +
            -2.90763742314713280951027603093668284e-2 * k[10] +
            3.51436748907427588126012411356312502e-2 * k[11] +
            5.03733798263124184187633780692175018e-2 * k[12] +
            1.32536591825484611547185585744510317e-1 * k[13] +
            8.12574984083043072171886210101269656e-2 * k[14] +
            -1.61565647687168908217617328536513442e-1 * k[15];
        Q_ptr[stride_Q + 4] =
            4.72672505780214059940355513139814428e-2 * k[0] +
            3.01680838086382395147509941989981593e-2 * k[3] +
            3.07141313567542534365442599499015746e-2 * k[4] +
            1.7826419382056305161067157242352542e-2 * k[5] +
            5.78742283883430451458273878060973433e-2 * k[6] +
            -3.4479093351646252414980954430454291e-2 * k[7] +
            9.52675169279657253591531972944255667e-3 * k[8] +
            -8.91186230019031152829638408404169546e-3 * k[10] +
            7.62244818287585731037944110922985951e-2 * k[11] +
            1.09256780549017683805064343868455495e-1 * k[12] +
            1.83524299335086263246100950008127766e-1 * k[13] +
            2.39292595637846304332057421606388371e-1 * k[14] +
            5.32393705945179176681195416984521418e-2 * k[15];
        Q_ptr[stride_Q + 5] =
            4.7448439553073359640550137527098737e-2 * k[0] +
            2.06809596688268504284247180163120666e-1 * k[3] +
            2.10552886249342824082599874586802258e-1 * k[4] +
            1.22204467018331660447316299202059757e-1 * k[5] +
            3.96741997521593084586825554780367414e-1 * k[6] +
            -2.36362622016760788292274510003904947e-1 * k[7] +
            6.53082140660319439726180351683260291e-2 * k[8] +
            -2.45265364966970699657678310144054469e-2 * k[10] +
            1.61130250547462903896224716783780329e-2 * k[11] +
            2.30956931441289236799592132548541268e-2 * k[12] +
            4.27668690250534650316916834407464267e-2 * k[13] +
            5.09096885301095810950507832241799055e-2 * k[14] +
            2.91335941627782210475611079923770418e-2 * k[15];
        Q_ptr[stride_Q + 6] =
            4.71556184862722217043176510883817568e-2 * k[0] +
            2.57505642984341518959643610103768758e-1 * k[3] +
            2.62166539774126204771386309576452771e-1 * k[4] +
            1.52160926567385574032313319916511754e-1 * k[5] +
            4.93996917003248424690717589322787684e-1 * k[6] +
            -2.94303117140325044155724474409270343e-1 * k[7] +
            8.13174723249510999973459944013676189e-2 * k[8];
    }

    this->load_back_from_temp();
}


// ########################################################################################################################
// Explicit Runge-Kutta Method of order 8(7) due to Verner
// ########################################################################################################################
/* Tableau, error weights, extra stages, and interpolant: Verner's exact rationals and 40-digit coefficients ("most
   efficient" pair RKV87.IIa, file RKV87.IIa.Efficient.000000282866.081208; the same values as OrdinaryDiffEq.jl). K
   holds the thirteen stages, then the derivative at the end of the step (Verner's stage 14), then the seven extra
   stages. */
CyrkErrorCodes Vern8::p_additional_setup() noexcept
{
    this->order     = Vern8_order;
    this->n_stages  = Vern8_n_stages;
    this->len_Pcols = Vern8_len_Pcols;
    this->error_estimator_order = Vern8_error_estimator_order;
    this->error_exponent        = Vern8_error_exponent;

    this->K_size     = this->n_stages + 1 + Vern8_nEXTRA_stages;
    this->integration_method = ODEMethod::VERN8;

    return RKSolver::p_additional_setup();
}

void Vern8::p_compute_stages() noexcept
{
    const size_t l_num_y                           = this->num_y;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    double* const CYRK_RESTRICT l_dy_old_ptr       = this->dy_old_ptr;
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;

    const double original_time = this->t_now;

    // ------------------------------------------------------------------------
    // Stage 1 (C = 1/20)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (1.0 / 20.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[0] = l_dy_old_ptr[y_i];

        const double temp_double =
            (1.0 / 20.0) * k[0];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 2 (C = 341/3200)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (341.0 / 3200.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[1] = l_dy_now_ptr[y_i];

        const double temp_double =
            (-7161.0 / 1024000.0) * k[0] +
            (116281.0 / 1024000.0) * k[1];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 3 (C = 1023/6400)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (1023.0 / 6400.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[2] = l_dy_now_ptr[y_i];

        const double temp_double =
            (1023.0 / 25600.0) * k[0] +
            (3069.0 / 25600.0) * k[2];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 4 (C = 39/100)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (39.0 / 100.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[3] = l_dy_now_ptr[y_i];

        const double temp_double =
            (4202367.0 / 11628100.0) * k[0] +
            (-3899844.0 / 2907025.0) * k[2] +
            (3982992.0 / 2907025.0) * k[3];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 5 (C = 93/200)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (93.0 / 200.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[4] = l_dy_now_ptr[y_i];

        const double temp_double =
            (5611.0 / 114400.0) * k[0] +
            (31744.0 / 135025.0) * k[3] +
            (923521.0 / 5106400.0) * k[4];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 6 (C = 31/200)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (31.0 / 200.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[5] = l_dy_now_ptr[y_i];

        const double temp_double =
            (21173.0 / 343200.0) * k[0] +
            (8602624.0 / 76559175.0) * k[3] +
            (-26782109.0 / 689364000.0) * k[4] +
            (5611.0 / 283500.0) * k[5];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 7 (C = 943/1000)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (943.0 / 1000.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[6] = l_dy_now_ptr[y_i];

        const double temp_double =
            -1.76763024022232687573559711957214559 * k[0] +
            (-125.0 / 2.0) * k[3] +
            -6.061889377376669100821361459659332 * k[4] +
            5.65082319822276313856129803060084017 * k[5] +
            6.56216964193762328379956605486306374e1 * k[6];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 8 (C = 0.901802)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (7067558016280.0 / 7837150160667.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[7] = l_dy_now_ptr[y_i];

        const double temp_double =
            -1.18094506655497079982511628262829796 * k[0] +
            -4.15047344111432084160664150270199423e1 * k[3] +
            -4.43443831910372501122516922984610021 * k[4] +
            4.26040818858613302481219371074469324 * k[5] +
            4.3753640224461715849876768294383793e1 * k[6] +
            7.87142548991231068744647504422630755e-3 * k[7];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 9 (C = 909/1000)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (909.0 / 1000.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[8] = l_dy_now_ptr[y_i];

        const double temp_double =
            -1.28140599944148840545951029118205425 * k[0] +
            -4.50471399601398663022075425713600732e1 * k[3] +
            -4.73136206944957647731146426549128281 * k[4] +
            4.514967016593807841185851584597241 * k[5] +
            4.74490955717298513486902239223592902e1 * k[6] +
            1.05922829711166113568739395551654288e-2 * k[7] +
            -5.74684226384461625443231847828629623e-3 * k[8];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 10 (C = 47/50)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + (47.0 / 50.0) * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[9] = l_dy_now_ptr[y_i];

        const double temp_double =
            -1.72447013426248519175670981748448186 * k[0] +
            -6.09234900848305401651843461925376525e1 * k[3] +
            -5.95151837622239245520283276706185487 * k[4] +
            5.5565237306984562359797916508435925 * k[5] +
            6.39830119803330533683753637863599594e1 * k[6] +
            1.46420282504149615927592139175945268e-2 * k[7] +
            6.46040877235820360362186514497765071e-2 * k[8] +
            -7.93032316900887898402445254869337329e-2 * k[9];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 11 (C = 1)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 1.0 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[10] = l_dy_now_ptr[y_i];

        const double temp_double =
            -3.30162266774707901635399478979098363 * k[0] +
            -1.18011272359752508566692330395789887e2 * k[3] +
            -1.0141422388456112486427839160345109e1 * k[4] +
            9.139311332232057923544012273556827 * k[5] +
            1.23375942828404268368484718098650189e2 * k[6] +
            4.62324437887458047483980762506763092 * k[7] +
            -3.38327773806820192365255097153681124 * k[8] +
            4.52759210032461818945126533935112904 * k[9] +
            -5.8284954858116229631930880191629857 * k[10];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Stage 12 (C = 1)
    // ------------------------------------------------------------------------
    this->t_now = this->t_old + 1.0 * this->step;
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[11] = l_dy_now_ptr[y_i];

        const double temp_double =
            -3.03951503376630903004010285182120025 * k[0] +
            -1.09260868089417625468644419232216462e2 * k[3] +
            -9.29064249740029344971766554265689755 * k[4] +
            8.4305049817649111421342992538361678 * k[5] +
            1.14201001037833131355742404109552343e2 * k[6] +
            -9.63727134214547935816237565898790165e-1 * k[7] +
            -5.03488408880218979119868033618333232 * k[8] +
            5.95813082400292317754040216538817207 * k[9];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Final Update (B Vector Dot Product)
    // ------------------------------------------------------------------------
    this->t_now = original_time;

    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[12] = l_dy_now_ptr[y_i];

        const double temp_double =
            4.42798941900795107471674666809851886e-2 * k[0] +
            3.54104939172444874481555202873356835e-1 * k[5] +
            2.47969215495643782866762941537066302e-1 * k[6] +
            -1.56942020388380840509920703427119121e1 * k[7] +
            2.50840649655585626134393003123718628e1 * k[8] +
            -3.17383677862602764683315611200729774e1 * k[9] +
            2.29382832739887839523148356034479702e1 * k[10] +
            -2.3613246330715421452599006412635176e-1 * k[11];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }

    // Derivative at the end of the step (k_13); it is also the first stage of the next step.
    this->diffeq(this);

    for (size_t y_i = 0; y_i < l_num_y; y_i++) {
        l_K_ptr_index_ptr[y_i][13] = l_dy_now_ptr[y_i];
    }
}

double Vern8::p_estimate_error() noexcept
{
    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    const double* const CYRK_RESTRICT l_rtols_ptr  = this->rtols_ptr;
    const double* const CYRK_RESTRICT l_atols_ptr  = this->atols_ptr;
    const bool l_use_array_rtols                   = this->use_array_rtols;
    const bool l_use_array_atols                   = this->use_array_atols;

    double rtol = l_rtols_ptr[0];
    double atol = l_atols_ptr[0];
    double l_error_norm = 0.0;

    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        rtol = l_use_array_rtols ? l_rtols_ptr[y_i] : rtol;
        atol = l_use_array_atols ? l_atols_ptr[y_i] : atol;

        const double* const k = l_K_ptr_index_ptr[y_i];

        // Difference between the propagated and embedded solutions (b - b_hat) dotted with K
        const double error_dot =
            -3.27210390102813776968984211051090278e-5 * k[0] +
            -5.04625061877770304762732216148668473e-4 * k[5] +
            1.21172358978475904764269386620436387e-4 * k[6] +
            -2.0142336771313868543717198659871561e1 * k[7] +
            5.23717859943982891412997631939498343 * k[8] +
            -8.15674440879465804863638151136902775 * k[9] +
            2.29382832739887839523148356034479702e1 * k[10] +
            -2.3613246330715421452599006412635176e-1 * k[11] +
            3.60167943728977516212453673774620241e-1 * k[12];

        const double scale = error_dot / (
            atol + std::max(std::abs(l_y_old_ptr[y_i]), std::abs(l_y_now_ptr[y_i])) * rtol);
        l_error_norm += (scale * scale);
    }

    return this->step_size * std::sqrt(l_error_norm) / this->num_y_sqrt;
}

void Vern8::set_Q_array(double* Q_ptr) noexcept
{
    // The extra stages overwrite the solver's current state, so hold a copy until they are done.
    this->offload_to_temp();

    double** const CYRK_RESTRICT l_K_ptr_index_ptr = this->K_ptr_index_ptr;
    double* const CYRK_RESTRICT l_y_now_ptr        = this->y_now_ptr;
    double* const CYRK_RESTRICT l_y_old_ptr        = this->y_old_ptr;
    double* const CYRK_RESTRICT l_dy_now_ptr       = this->dy_now_ptr;
    const size_t l_num_y                           = this->num_y;

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 15, stored in K[14] (C = 0.311018)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];

        const double temp_double =
            4.62070064675496310173041315023811643e-2 * k[0] +
            4.5039041608424808668285203844006797e-2 * k[5] +
            2.33681669771342441078870106534022113e-1 * k[6] +
            3.78390136842106741078033822086185525e1 * k[7] +
            -1.59491132894542461026613949030739737e1 * k[8] +
            2.30283683518161028514251059632959009e1 * k[9] +
            -4.485578507769412524816130998016948e1 * k[10] +
            -6.37985876864744400950906740233014078e-2 * k[11] +
            -1.25950355438616626824103246451984216e-2 * k[13];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + 3.1101776349538638639274173188290997e-1 * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 16, stored in K[15] (C = 69/400)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[14] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.03794685548204099306515874722069611e-2 * k[0] +
            4.10983613104607933991653061402884825e-2 * k[5] +
            1.71805415334819578329630920954942462e-1 * k[6] +
            4.61410531998151886974342237185977125 * k[7] +
            -1.79166788308539644971274499674683647 * k[8] +
            2.53165893048504140846224351879291361 * k[9] +
            -5.3249778602057307192571881597727627 * k[10] +
            -3.06553259538563473492444949635651311e-2 * k[11] +
            -5.25447997942961357054951909437787811e-3 * k[13] +
            -8.3991946442247929975386534642580587e-2 * k[14];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (69.0 / 400.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 17, stored in K[16] (C = 3923/5000)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[15] = l_dy_now_ptr[y_i];

        const double temp_double =
            4.0828971329970796202071187562426538e-2 * k[0] +
            4.24447951424763221889208665773233249e-1 * k[5] +
            2.32609153127523453946510009696484549e-1 * k[6] +
            2.67798252071180606278052887101403596 * k[7] +
            7.42082665733894521647760704402296362e-1 * k[8] +
            1.4603778479414611939209923397313123e-1 * k[9] +
            -3.57934450989056521803335674382591768 * k[10] +
            1.13884438960017370453163871614998567e-1 * k[11] +
            1.26779065103319004737869353761568723e-2 * k[13] +
            -7.44343634994667442975278503256155248e-2 * k[14] +
            4.78274807975785155457551147387698766e-2 * k[15];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (3923.0 / 5000.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 18, stored in K[17] (C = 37/100)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[16] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.21268239366841362992813692799451468e-2 * k[0] +
            5.39250839674479771820910686234706563e-2 * k[5] +
            1.6607580974346408285419305999282519e-2 * k[6] +
            -4.45448575792677965541893699329846307 * k[7] +
            6.83521827863214638171129681796815263 * k[8] +
            -8.71133482218199373984717273484883797 * k[9] +
            6.49163583923291705365126714270310565 * k[10] +
            -7.07255180984434642206998522770029465e-2 * k[11] +
            -1.85403149199321642911184293794120297e-2 * k[13] +
            2.35040210543538464511654208704596219e-2 * k[14] +
            2.34479510340782209055637781340277478e-1 * k[15] +
            -8.24107250115289888582308969809776877e-2 * k[16];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (37.0 / 100.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 19, stored in K[18] (C = 1/2)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[17] = l_dy_now_ptr[y_i];

        const double temp_double =
            5.02010287035571359869996441997788346e-2 * k[0] +
            1.55220903479549811493222610470056764e-1 * k[5] +
            1.26426842408923491471309113486474751e-1 * k[6] +
            -5.14920630353984701704917414605721855 * k[7] +
            8.46834099903692926607453176331494312 * k[8] +
            -1.0662130681081495275442098362070955e1 * k[9] +
            7.54183322495972836290996201569018334 * k[10] +
            -7.43696811383214243944066492459357054e-2 * k[11] +
            -2.05588768661838261933982175922112176e-2 * k[13] +
            7.75379526471029807261782993777862396e-2 * k[14] +
            1.04625922035254429631376197133398759e-1 * k[15] +
            -1.17921330645197935214502268706301346e-1 * k[16];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (1.0 / 2.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 20, stored in K[19] (C = 7/10)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[18] = l_dy_now_ptr[y_i];

        const double temp_double =
            3.73734144645782569275750654880009413e-2 * k[0] +
            3.50493070533831640676708746833907109e-1 * k[5] +
            4.92265281937302543329898982417348481e-1 * k[6] +
            8.55369543935931224228430442172531586 * k[7] +
            -1.0353172990305913485325740067192078e1 * k[8] +
            1.38332042725291499035108287546054477e1 * k[9] +
            -1.22809243307846186372952358378451905e1 * k[10] +
            1.71915159565650976274681011337864431e-1 * k[11] +
            3.64158311431449638011382238421452822e-2 * k[13] +
            2.96192058028876305489014641252072343e-2 * k[14] +
            -2.65179393862706700264761562373842503e-1 * k[15] +
            9.42950396173806655317007970358739476e-2 * k[16];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (7.0 / 10.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Extra Stage: Verner stage 21, stored in K[20] (C = 9/10)
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[19] = l_dy_now_ptr[y_i];

        const double temp_double =
            3.93905834552825094341067063492352199e-2 * k[0] +
            3.55851614123442418313669732275532372e-1 * k[5] +
            4.19738222595261002937222552672006537e-1 * k[6] +
            8.72044977807194166293172525204036071e-1 * k[7] +
            8.98952083487659486126627160171417044e-1 * k[8] +
            -6.3058061610598835902345664952785347e-1 * k[9] +
            -1.12188722059548355073668164542521508 * k[10] +
            4.29821951240019717696751103182919771e-2 * k[11] +
            1.33255756687391570701349589188919056e-2 * k[13] +
            1.87622705396414803444610129192809777e-2 * k[14] +
            -1.85941113292210557051537936859259651e-1 * k[15] +
            1.773614271924602745226064729836361e-1 * k[16];

        l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
    }
    this->t_now = this->t_old + (9.0 / 10.0) * this->step;
    this->diffeq(this);

    // ------------------------------------------------------------------------
    // Build Dense Interpolator (Q Matrix): Q[j] = sum_i b_i(theta_j) k_i at the nodes Vern8_dense_nodes
    // ------------------------------------------------------------------------
    for (size_t y_i = 0; y_i < l_num_y; y_i++)
    {
        double* const k = l_K_ptr_index_ptr[y_i];
        k[20] = l_dy_now_ptr[y_i];

        const size_t stride_Q = y_i * 8;

        Q_ptr[stride_Q + 0] =
            2.61711657486117840238947211373106868e-2 * k[0] +
            1.56300163839692302953479369798281718e-1 * k[5] +
            1.09452381827113087590209000906287563e-1 * k[6] +
            -6.9273429388937179924471147940900172 * k[7] +
            1.10719818623399163556359335287672657e1 * k[8] +
            -1.4009158122985361683261445822187651e1 * k[9] +
            1.01248444664584104173983924290849789e1 * k[10] +
            -1.04227698119750886595935431624061434e-1 * k[11] +
            -1.46497881497256606954251354987031357e-2 * k[13] +
            1.37324954389864987728856298536324971e-2 * k[14] +
            -9.81347021194943605262853213667869635e-2 * k[15] +
            -3.64919756479812893283208313739143299e-2 * k[16] +
            -6.76550803524988803425134940740911137e-2 * k[17] +
            -8.73398384323045183403616136405674635e-2 * k[18] +
            -6.06271732057640954278002336416869353e-2 * k[19] +
            -5.87692802461310794095920020502775123e-2 * k[20];
        Q_ptr[stride_Q + 1] =
            4.14128891134176149355251133727423251e-2 * k[0] +
            6.65978511832714989649407685030958226e-1 * k[5] +
            4.66365054105309705937388855151050793e-1 * k[6] +
            -2.95166775777091008456640997722272751e1 * k[7] +
            4.7176546860710352895954208271742172e1 * k[8] +
            -5.96915450987242479023796320167702856e1 * k[9] +
            4.31408943193782495657618058586302469e1 * k[10] +
            -4.44103227919417457717758773672302491e-1 * k[11] +
            -6.12023592273507868647113599008239377e-2 * k[13] +
            4.83609040626014392872294520668132105e-2 * k[14] +
            -3.45595083974271316877973441330023011e-1 * k[15] +
            -1.28511597998175471250757903667242185e-1 * k[16] +
            -3.1596044853914925763806678418711395e-1 * k[17] +
            -3.40716334315575578648276859172221824e-1 * k[18] +
            -2.90581589861012936698499828523859615e-1 * k[19] +
            -2.58180845934344657785788496542835711e-1 * k[20];
        Q_ptr[stride_Q + 2] =
            4.43888872460360886392512420983397064e-2 * k[0] +
            3.58118327083866377578634933791319116e-1 * k[5] +
            2.50779672345527600908955169574572634e-1 * k[6] +
            -1.58720784640844588155004522188716515e1 * k[7] +
            2.53683651036401110213226609407171347e1 * k[8] +
            -3.20980870883951582862054473804156519e1 * k[9] +
            2.31982633494312120858239105851108843e1 * k[10] +
            -2.38808763660224410168011902122499384e-1 * k[11] +
            -2.87664924840303701752172126030827003e-2 * k[13] +
            8.07916072911480749636391026973771007e-4 * k[14] +
            -5.77350296637501032200877758092945527e-3 * k[15] +
            -2.14691159296495123081165512237153349e-3 * k[16] +
            -1.21226759075053438757521049777046018e-1 * k[17] +
            -1.6045157464003258124528796826532518e-1 * k[18] +
            -2.22671793574431012208837672719703473e-1 * k[19] +
            -1.62118155346935779209453424840963062e-1 * k[20];
        Q_ptr[stride_Q + 3] =
            4.42099657591557987493756813683150445e-2 * k[0] +
            3.66123450787823402544158953479660621e-1 * k[5] +
            2.56385423706846858247821429992705802e-1 * k[6] +
            -1.62268716760893643598135609346473642e1 * k[7] +
            2.59354315882722361209819880803642211e1 * k[8] +
            -3.2815585016790247254500073474640341e1 * k[9] +
            2.37168209148631530767681767204399925e1 * k[10] +
            -2.44146925798576634435698949111556208e-1 * k[11] +
            -2.96476479219302434139768422255166859e-2 * k[13] +
            1.49048830561720986876210508194051781e-3 * k[14] +
            -1.06512779512075226123612945321364035e-2 * k[15] +
            -3.96074138118907923361100112541749754e-3 * k[16] +
            1.16031403153455179021377433671239755e-2 * k[17] +
            -1.08150323490692499072845346405623028e-1 * k[18] +
            -2.28434903100920531177596406170480896e-1 * k[19] +
            -1.64616459486049860802696465235523512e-1 * k[20];
        Q_ptr[stride_Q + 4] =
            4.64622898907344024599955205078658342e-2 * k[0] +
            2.03328597213125953145586957272128364e-1 * k[5] +
            1.42385002752573924018666615089975841e-1 * k[6] +
            -9.01167911521931930678613594213562529 * k[7] +
            1.44033792744307138313829666998359547e1 * k[8] +
            -1.82243089150244219369823911206703613e1 * k[9] +
            1.31712620882312315996358457472086985e1 * k[10] +
            -1.35588288129870161131626461553806455e-1 * k[11] +
            -1.20446934804842106929322067667357226e-2 * k[13] +
            -1.66196493994755500446111583324928795e-2 * k[14] +
            1.18766785715993415989001356447114581e-1 * k[15] +
            4.41641392752145738739745865768853152e-2 * k[16] +
            1.66819276951260233791684048650482224e-2 * k[17] +
            1.54886797208638886693358719281660304e-1 * k[18] +
            -9.54082540524833375668738027266599914e-2 * k[19] +
            -1.14261737107298107373993914899649894e-1 * k[20];
        Q_ptr[stride_Q + 5] =
            4.27401106410143107988894919104602585e-2 * k[0] +
            4.52168735361715396298529319952705239e-1 * k[5] +
            3.16640391521618730628846278331577213e-1 * k[6] +
            -2.00404645724435841531178886829461431e1 * k[7] +
            3.20307024231712166720510118929860054e1 * k[8] +
            -4.05278098009514145719779207287342184e1 * k[9] +
            2.92906802249297895104984414269390053e1 * k[10] +
            -3.01525636894451886108812579717896086e-1 * k[11] +
            -4.33229803891262031801602142756405599e-2 * k[13] +
            1.06593785582103104754793796710908862e-2 * k[14] +
            -7.61736964877591272751055768567905167e-2 * k[15] +
            -2.83256444174382796355827652712901321e-2 * k[16] +
            -1.76188425482777260859116954535807475e-2 * k[17] +
            -8.49137879824245841007646508808404739e-2 * k[18] +
            -1.52689199032957249768881152999164582e-2 * k[19] +
            -1.54651757165792674292162780354527473e-1 * k[20];
        Q_ptr[stride_Q + 6] =
            4.33444851941454275080511774709574768e-2 * k[0] +
            4.13956816812287302357323695194131487e-1 * k[5] +
            2.8988171516022879128222299418871498e-1 * k[6] +
            -1.8346883083837745340082611548780601e1 * k[7] +
            2.93238487724072447437721735863102961e1 * k[8] +
            -3.71028817906991801408765470888384441e1 * k[9] +
            2.68153806310359142402543519201934566e1 * k[10] +
            -2.76044279656521167947329024901839591e-1 * k[11] +
            -3.16464136124202783045584853451435486e-2 * k[13] +
            6.51123637683343854811971079181384387e-3 * k[14] +
            -4.65303808116410210649184305180748897e-2 * k[15] +
            -1.73026002708209869138513310082759078e-2 * k[16] +
            -1.0511450298909835797551918587361114e-2 * k[17] +
            -5.23755765693044946649487671199366851e-2 * k[18] +
            -7.89444989391392081176288802764888669e-3 * k[19] +
            -3.89395688361967572581636010220442013e-2 * k[20];
        Q_ptr[stride_Q + 7] =
            4.42798941900795107471674666809851886e-2 * k[0] +
            3.54104939172444874481555202873356835e-1 * k[5] +
            2.47969215495643782866762941537066302e-1 * k[6] +
            -1.56942020388380840509920703427119121e1 * k[7] +
            2.50840649655585626134393003123718628e1 * k[8] +
            -3.17383677862602764683315611200729774e1 * k[9] +
            2.29382832739887839523148356034479702e1 * k[10] +
            -2.3613246330715421452599006412635176e-1 * k[11];
    }

    this->load_back_from_temp();
}
