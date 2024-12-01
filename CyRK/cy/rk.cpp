#include "rk.hpp"
#include "dense.hpp"
#include "cysolution.hpp"

// ########################################################################################################################
// RKSolver (Base)
// ########################################################################################################################
// Constructors
RKSolver::RKSolver() {}
RKSolver::RKSolver(
    // Base Class input arguments
    DiffeqFuncType diffeq_ptr,
    std::shared_ptr<CySolverResult> storage_sptr,
    const double t_start,
    const double t_end,
    const double* const y0_ptr,
    const size_t num_y,
    const size_t num_extra,
    const char* args_ptr,
    const size_t size_of_args,
    const size_t max_num_steps,
    const size_t max_ram_MB,
    const bool use_dense_output,
    const double* t_eval,
    const size_t len_t_eval,
    PreEvalFunc pre_eval_func,
    // RKSolver input arguments
    const double rtol,
    const double atol,
    const double* rtols_ptr,
    const double* atols_ptr,
    const double max_step_size,
    const double first_step_size) :
        CySolverBase(
            diffeq_ptr,
            storage_sptr,
            t_start,
            t_end,
            y0_ptr,
            num_y,
            num_extra,
            args_ptr,
            size_of_args,
            max_num_steps,
            max_ram_MB,
            use_dense_output,
            t_eval,
            len_t_eval,
            pre_eval_func),
        user_provided_first_step_size(first_step_size),
        max_step_size(max_step_size)        
{
    // Check for errors
    if (this->user_provided_first_step_size != 0.0) [[unlikely]]
    {
        if (this->user_provided_first_step_size < 0.0) [[unlikely]]
        {
            this->storage_sptr->error_code = -1;
            this->storage_sptr->update_message("User-provided initial step size must be a positive number.");
        }
        else if (first_step_size > (this->t_delta_abs * 0.5)) [[unlikely]]
        {
            this->storage_sptr->error_code = -1;
            this->storage_sptr->update_message("User-provided initial step size must be smaller than 50 % of the time span size.");
        }
    }

    // Setup tolerances
    // User can provide an array of relative tolerances, one for each y value.
    // The length of the pointer array must be the same as y0 (and <= 25).
    if (rtols_ptr)
    {
        // rtol for each y
        this->use_array_rtols = true;
        // Allocate array
        this->rtols.resize(this->num_y);
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            double temp_double = rtols_ptr[y_i];
            if (temp_double < EPS_100) [[unlikely]]
            {
                temp_double = EPS_100;
            }
            this->rtols[y_i] = temp_double;
        }
    }
    else {
        // only one rtol
        double temp_double = rtol;
        this->rtols.resize(1);
        if (temp_double < EPS_100) [[unlikely]]
        {
            temp_double = EPS_100;
        }
        this->rtols[0] = temp_double;
    }

    if (atols_ptr)
    {
        // atol for each y
        this->use_array_atols = true;
        this->atols.resize(this->num_y);
        std::memcpy(this->atols.data(), atols_ptr, sizeof(double) * this->num_y);
    }
    else {
        // only one atol
        this->atols.resize(1);
        this->atols[0] = atol;
    }

    // Setup rtol and atol pointers
    this->rtols_ptr = this->rtols.data();
    this->atols_ptr = this->atols.data();
}

// Protected Methods
void RKSolver::p_estimate_error()
{   
    // Initialize rtol and atol
    double rtol = this->rtols_ptr[0];
    double atol = this->atols_ptr[0];

    this->error_norm = 0.0;

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

        // Dot product between K and E
        double error_dot      = 0.0;
        const size_t stride_K = y_i * this->n_stages_p1;

        switch (this->n_stages)
        {
        // These loops go 1 more than `n_stages`.
        // Note: DOP853 is handled in an override by its subclass.
        case(3):
            // RK23
            error_dot += this->E_ptr[0] * this->K_ptr[stride_K];
            error_dot += this->E_ptr[1] * this->K_ptr[stride_K + 1];
            error_dot += this->E_ptr[2] * this->K_ptr[stride_K + 2];
            error_dot += this->E_ptr[3] * this->K_ptr[stride_K + 3];

            break;
        case(6):
            // RK45
            error_dot += this->E_ptr[0] * this->K_ptr[stride_K];
            error_dot += this->E_ptr[1] * this->K_ptr[stride_K + 1];
            error_dot += this->E_ptr[2] * this->K_ptr[stride_K + 2];
            error_dot += this->E_ptr[3] * this->K_ptr[stride_K + 3];
            error_dot += this->E_ptr[4] * this->K_ptr[stride_K + 4];
            error_dot += this->E_ptr[5] * this->K_ptr[stride_K + 5];
            error_dot += this->E_ptr[6] * this->K_ptr[stride_K + 6];

            break;
        [[unlikely]] default:
            // Resort to unrolled loops
            // New or Non-optimized RK method. default to for loop.
            for (size_t j = 0; j < this->n_stages_p1; j++)
            {
                error_dot += this->E_ptr[j] * this->K_ptr[stride_K + j];
            }
            break;
        }

        // Find scale of y for error calculations
        const double scale_inv = 1.0 / (atol + std::fmax(std::fabs(this->y_old[y_i]), std::fabs(this->y_now[y_i])) * rtol);

        // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
        // TODO: This will need to change if CySolver ever accepts complex numbers
        // error_norm_abs = fabs(error_dot_1)
        error_dot *= scale_inv;

        this->error_norm += (error_dot * error_dot);
    }
    this->error_norm = this->step_size * std::sqrt(this->error_norm) / this->num_y_sqrt;
}


    //     double error_norm_abs = std::fabs(error_dot) * scale_inv * this->step;

    //     this->error_norm += (error_norm_abs * error_norm_abs);
    // }
    // this->error_norm = std::sqrt(this->error_norm) / this->num_y_sqrt;


void RKSolver::p_step_implementation()
{
    // Run RK integration step
    
    // Create local variables instead of calling class attributes for pointer objects.
    double* const l_K_ptr       = this->K_ptr;
    const double* const l_A_ptr = this->A_ptr;
    const double* const l_B_ptr = this->B_ptr;
    const double* const l_C_ptr = this->C_ptr;
    double* const l_y_now_ptr   = this->y_now.data();
    double* const l_y_old_ptr   = this->y_old.data();
    double* const l_dy_now_ptr  = this->dy_now.data();
    double* const l_dy_old_ptr  = this->dy_old.data();

    // Determine step size based on previous loop
    // Find minimum step size based on the value of t (less floating point numbers between numbers when t is large)
    const double min_step_size = 10. * std::fabs(std::nextafter(this->t_old, this->direction_inf) - this->t_old);
    // Look for over/undershoots in previous step size
    this->step_size = std::min<double>(this->step_size, this->max_step_size);
    this->step_size = std::max<double>(this->step_size, min_step_size);

    // Optimization variables
    // Define a very specific A (Row 1; Col 0) now since it is called consistently and does not change.
    const double A_at_10 = l_A_ptr[1 * this->len_Acols + 0];

    // Determine new step size
    bool step_accepted = false;
    bool step_rejected = false;
    bool step_error    = false;

    // !! Step Loop
    while (!step_accepted) {

        // Check if step size is too small
        // This will cause integration to fail: step size smaller than spacing between numbers
        if (this->step_size < min_step_size) [[unlikely]] {
            step_error = true;
            this->status = -1;
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
            const size_t stride_A = s * this->len_Acols;

            for (size_t y_i = 0; y_i < this->num_y; y_i++)
            {
                double temp_double;
                const size_t stride_K = y_i * this->n_stages_p1;
                // Dot Product (K, a) * step
                switch (s)
                {
                case(1):
                    // Set the first column of K
                    temp_double = l_dy_old_ptr[y_i];
                    // K[0, :] == first part of the array
                    l_K_ptr[stride_K] = temp_double;
                    temp_double *= A_at_10;
                    break;
                case(2):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    break;
                case(3):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    break;
                case(4):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3] * l_K_ptr[stride_K + 3];
                    break;
                case(5):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3] * l_K_ptr[stride_K + 3];
                    // j = 4
                    temp_double += l_A_ptr[stride_A + 4] * l_K_ptr[stride_K + 4];
                    break;
                case(6):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3] * l_K_ptr[stride_K + 3];
                    // j = 4
                    temp_double += l_A_ptr[stride_A + 4] * l_K_ptr[stride_K + 4];
                    // j = 5
                    temp_double += l_A_ptr[stride_A + 5] * l_K_ptr[stride_K + 5];
                    break;
                case(7):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3] * l_K_ptr[stride_K + 3];
                    // j = 4
                    temp_double += l_A_ptr[stride_A + 4] * l_K_ptr[stride_K + 4];
                    // j = 5
                    temp_double += l_A_ptr[stride_A + 5] * l_K_ptr[stride_K + 5];
                    // j = 6
                    temp_double += l_A_ptr[stride_A + 6] * l_K_ptr[stride_K + 6];
                    break;
                case(8):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3] * l_K_ptr[stride_K + 3];
                    // j = 4
                    temp_double += l_A_ptr[stride_A + 4] * l_K_ptr[stride_K + 4];
                    // j = 5
                    temp_double += l_A_ptr[stride_A + 5] * l_K_ptr[stride_K + 5];
                    // j = 6
                    temp_double += l_A_ptr[stride_A + 6] * l_K_ptr[stride_K + 6];
                    // j = 7
                    temp_double += l_A_ptr[stride_A + 7] * l_K_ptr[stride_K + 7];
                    break;
                case(9):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3] * l_K_ptr[stride_K + 3];
                    // j = 4
                    temp_double += l_A_ptr[stride_A + 4] * l_K_ptr[stride_K + 4];
                    // j = 5
                    temp_double += l_A_ptr[stride_A + 5] * l_K_ptr[stride_K + 5];
                    // j = 6
                    temp_double += l_A_ptr[stride_A + 6] * l_K_ptr[stride_K + 6];
                    // j = 7
                    temp_double += l_A_ptr[stride_A + 7] * l_K_ptr[stride_K + 7];
                    // j = 8
                    temp_double += l_A_ptr[stride_A + 8] * l_K_ptr[stride_K + 8];
                    break;
                case(10):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]     * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1] * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2] * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3] * l_K_ptr[stride_K + 3];
                    // j = 4
                    temp_double += l_A_ptr[stride_A + 4] * l_K_ptr[stride_K + 4];
                    // j = 5
                    temp_double += l_A_ptr[stride_A + 5] * l_K_ptr[stride_K + 5];
                    // j = 6
                    temp_double += l_A_ptr[stride_A + 6] * l_K_ptr[stride_K + 6];
                    // j = 7
                    temp_double += l_A_ptr[stride_A + 7] * l_K_ptr[stride_K + 7];
                    // j = 8
                    temp_double += l_A_ptr[stride_A + 8] * l_K_ptr[stride_K + 8];
                    // j = 9
                    temp_double += l_A_ptr[stride_A + 9] * l_K_ptr[stride_K + 9];
                    break;
                case(11):
                    // Loop through (j = 0; j < s; j++)
                    // j = 0
                    temp_double  = l_A_ptr[stride_A]      * l_K_ptr[stride_K];
                    // j = 1
                    temp_double += l_A_ptr[stride_A + 1]  * l_K_ptr[stride_K + 1];
                    // j = 2
                    temp_double += l_A_ptr[stride_A + 2]  * l_K_ptr[stride_K + 2];
                    // j = 3
                    temp_double += l_A_ptr[stride_A + 3]  * l_K_ptr[stride_K + 3];
                    // j = 4
                    temp_double += l_A_ptr[stride_A + 4]  * l_K_ptr[stride_K + 4];
                    // j = 5
                    temp_double += l_A_ptr[stride_A + 5]  * l_K_ptr[stride_K + 5];
                    // j = 6
                    temp_double += l_A_ptr[stride_A + 6]  * l_K_ptr[stride_K + 6];
                    // j = 7
                    temp_double += l_A_ptr[stride_A + 7]  * l_K_ptr[stride_K + 7];
                    // j = 8
                    temp_double += l_A_ptr[stride_A + 8]  * l_K_ptr[stride_K + 8];
                    // j = 9
                    temp_double += l_A_ptr[stride_A + 9]  * l_K_ptr[stride_K + 9];
                    // j = 10
                    temp_double += l_A_ptr[stride_A + 10] * l_K_ptr[stride_K + 10];
                    break;
                [[unlikely]] default:
                    // Resort to regular rolled loops
                    // Initialize
                    temp_double = 0.0;

                    for (size_t j = 0; j < s; j++)
                    {
                        temp_double += l_A_ptr[stride_A + j] * l_K_ptr[stride_K + j];
                    }
                    break;
                }
                // Update value of y_now
                l_y_now_ptr[y_i] = l_y_old_ptr[y_i] + (this->step * temp_double);
            }
            // Call diffeq method to update K with the new dydt
            // This will use the now updated dy_now_ptr based on the values of y_now_ptr and t_now_ptr.
            this->diffeq(this);

            // Update K based on the new dy values.
            for (size_t y_i = 0; y_i < this->num_y; y_i++) {
                const size_t stride_K = y_i * this->n_stages_p1;
                l_K_ptr[stride_K + s] = l_dy_now_ptr[y_i];
            }
        }

        // Restore t_now to its previous value.
        this->t_now = time_tmp;

        // Dot Product (K, B) * step
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            double temp_double;
            const size_t stride_K = y_i * this->n_stages_p1;

            switch (this->n_stages)
            {
            case(3):
                // RK23
                temp_double  = l_B_ptr[0] * l_K_ptr[stride_K];
                temp_double += l_B_ptr[1] * l_K_ptr[stride_K + 1];
                temp_double += l_B_ptr[2] * l_K_ptr[stride_K + 2];
                break;
            case(6):
                //RK45
                temp_double  = l_B_ptr[0] * l_K_ptr[stride_K];
                temp_double += l_B_ptr[1] * l_K_ptr[stride_K + 1];
                temp_double += l_B_ptr[2] * l_K_ptr[stride_K + 2];
                temp_double += l_B_ptr[3] * l_K_ptr[stride_K + 3];
                temp_double += l_B_ptr[4] * l_K_ptr[stride_K + 4];
                temp_double += l_B_ptr[5] * l_K_ptr[stride_K + 5];
                break;
            case(12):
                //DOP853
                temp_double  = l_B_ptr[0]  * l_K_ptr[stride_K];
                temp_double += l_B_ptr[1]  * l_K_ptr[stride_K + 1];
                temp_double += l_B_ptr[2]  * l_K_ptr[stride_K + 2];
                temp_double += l_B_ptr[3]  * l_K_ptr[stride_K + 3];
                temp_double += l_B_ptr[4]  * l_K_ptr[stride_K + 4];
                temp_double += l_B_ptr[5]  * l_K_ptr[stride_K + 5];
                temp_double += l_B_ptr[6]  * l_K_ptr[stride_K + 6];
                temp_double += l_B_ptr[7]  * l_K_ptr[stride_K + 7];
                temp_double += l_B_ptr[8]  * l_K_ptr[stride_K + 8];
                temp_double += l_B_ptr[9]  * l_K_ptr[stride_K + 9];
                temp_double += l_B_ptr[10] * l_K_ptr[stride_K + 10];
                temp_double += l_B_ptr[11] * l_K_ptr[stride_K + 11];
                break;
            [[unlikely]] default:
                // Resort to rolled loops
                // Initialize
                temp_double = 0.0;

                for (size_t j = 0; j < this->n_stages; j++)
                {
                    temp_double += l_B_ptr[j] * l_K_ptr[stride_K + j];
                }
                break;
            }
            // Update y_now
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
        this->status = -1;
    }
    else if (!step_accepted) [[unlikely]]
    {
        // Issue with step convergence
        this->status = -7;
    }

    // End of RK step.
}


// Public methods
void RKSolver::reset()
{
    // Update stride information
    this->nstages_numy = this->n_stages * this->num_y;
    this->n_stages_p1  = this->n_stages + 1;

    // It is important to initialize the K variable with zeros
    std::fill(this->K_ptr, this->K_ptr + (this->num_y * this->n_stages_p1), 0.0);

    // Call base class reset after K is established but before first step size is calculated.
    CySolverBase::reset();

    // Update initial step size
    if (this->user_provided_first_step_size == 0.0) [[likely]]
    {
        // User did not provide a step size. Try to find a good guess.
        this->calc_first_step_size();
    }
    else {
        this->step_size = this->user_provided_first_step_size;
    }
}

void RKSolver::calc_first_step_size()
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

            y_old_tmp = this->y_old[y_i];
            scale     = atol + std::fabs(y_old_tmp) * rtol;
            d0_abs    = std::fabs(y_old_tmp / scale);
            d1_abs    = std::fabs(this->dy_old[y_i] / scale);
            d0       += (d0_abs * d0_abs);
            d1       += (d1_abs * d1_abs);
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
            this->y_now[y_i] = this->y_old[y_i] + h0_direction * this->dy_old[y_i];
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

            scale  = atol + std::fabs(this->y_old[y_i]) * rtol;
            d2_abs = std::fabs((this->dy_now[y_i] - this->dy_old[y_i]) / scale);
            d2    += (d2_abs * d2_abs);
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

    case(0):
        // RK23
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
            temp_double = this->K_ptr[stride_K] * this->P_ptr[stride_P];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[stride_P + 1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[stride_P + 2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[stride_P + 3];
            Q_ptr[stride_Q + 1] = temp_double;

            // P = 2
            stride_P += this->n_stages_p1;
            temp_double = this->K_ptr[stride_K] * this->P_ptr[stride_P];
            temp_double += this->K_ptr[stride_K + 1] * this->P_ptr[stride_P + 1];
            temp_double += this->K_ptr[stride_K + 2] * this->P_ptr[stride_P + 2];
            temp_double += this->K_ptr[stride_K + 3] * this->P_ptr[stride_P + 3];
            Q_ptr[stride_Q + 2] = temp_double;
        }
        break;

    case(1):
        // RK45
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

    case(2):
        {
        // DOP853
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
            temp_double   = 0.0;
            temp_double_array_ptr[y_i * 2]     = 0.0;
            temp_double_array_ptr[y_i * 2 + 1] = 0.0;
            for (size_t n_i = 0; n_i < this->n_stages_p1; n_i++)
            {
                K_ni = K_ptr[stride_K + n_i];
                temp_double                  += K_ni * DOP853_AEXTRA_ptr[3 * n_i];
                temp_double_array_ptr[y_i * 2]     += K_ni * DOP853_AEXTRA_ptr[3 * n_i + 1];
                temp_double_array_ptr[y_i * 2 + 1] += K_ni * DOP853_AEXTRA_ptr[3 * n_i + 2];
            }

            // Update y for diffeq call using the temp_double dot product.
            this->y_now[y_i] = this->y_old[y_i] + temp_double * this->step;
        }
        // Update time and call the diffeq.
        this->t_now = this->t_old + (this->step * DOP853_CEXTRA_ptr[0]);
        this->diffeq(this);

        // S (row) == 14
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            // Store dy from the last call.
            K_ni = this->dy_now[y_i];
            K_extended_ptr[y_i * 3] = K_ni;

            // Dot Product (K.T dot a) * h
            // Add row 14 to the remaining dot product trackers
            temp_double_array_ptr[y_i * 2]     += K_ni * DOP853_AEXTRA_ptr[3 * 13 + 1];
            temp_double_array_ptr[y_i * 2 + 1] += K_ni * DOP853_AEXTRA_ptr[3 * 13 + 2];

            // Update y for diffeq call
            this->y_now[y_i] = this->y_old[y_i] + temp_double_array_ptr[y_i * 2] * this->step;
        }
        // Update time and call the diffeq.
        this->t_now = this->t_old + (this->step * DOP853_CEXTRA_ptr[1]);
        this->diffeq(this);

        // S (row) == 15
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            // Store dy from the last call.
            K_ni = this->dy_now[y_i];
            K_extended_ptr[y_i * 3 + 1] = K_ni;

            // Dot Product (K.T dot a) * h            
            // Add row 15 to the remaining dot product trackers
            temp_double_array_ptr[y_i * 2 + 1] += K_ni * DOP853_AEXTRA_ptr[3 * 14 + 2];

            // Update y for diffeq call
            this->y_now[y_i] = this->y_old[y_i] + temp_double_array_ptr[y_i * 2 + 1] * this->step;
        }
        // Update time and call the diffeq.
        this->t_now = this->t_old + (this->step * DOP853_CEXTRA_ptr[2]);
        this->diffeq(this);


        // Done with diffeq calls. Now build up Q matrix.
        for (size_t y_i = 0; y_i < this->num_y; y_i++)
        {
            // Store dy from the last call.
            K_extended_ptr[y_i * 3 + 2] = this->dy_now[y_i];

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
                temp_double   += K_ni * DOP853_D_ptr[4 * n_i];
                // Row 2
                temp_double_2 += K_ni * DOP853_D_ptr[4 * n_i + 1];
                // Row 3
                temp_double_3 += K_ni * DOP853_D_ptr[4 * n_i + 2];
                // Row 4
                temp_double_4 += K_ni * DOP853_D_ptr[4 * n_i + 3];
            }
            // Now add the extra 3 rows from extended
            // Row 1
            temp_double   += K_extended_ptr[y_i * 3]     * DOP853_D_ptr[4 * 13];
            temp_double_2 += K_extended_ptr[y_i * 3]     * DOP853_D_ptr[4 * 13 + 1];
            temp_double_3 += K_extended_ptr[y_i * 3]     * DOP853_D_ptr[4 * 13 + 2];
            temp_double_4 += K_extended_ptr[y_i * 3]     * DOP853_D_ptr[4 * 13 + 3];
            // Row 2
            temp_double   += K_extended_ptr[y_i * 3 + 1] * DOP853_D_ptr[4 * 14];
            temp_double_2 += K_extended_ptr[y_i * 3 + 1] * DOP853_D_ptr[4 * 14 + 1];
            temp_double_3 += K_extended_ptr[y_i * 3 + 1] * DOP853_D_ptr[4 * 14 + 2];
            temp_double_4 += K_extended_ptr[y_i * 3 + 1] * DOP853_D_ptr[4 * 14 + 3];
            // Row 3
            temp_double   += K_extended_ptr[y_i * 3 + 2] * DOP853_D_ptr[4 * 15];
            temp_double_2 += K_extended_ptr[y_i * 3 + 2] * DOP853_D_ptr[4 * 15 + 1];
            temp_double_3 += K_extended_ptr[y_i * 3 + 2] * DOP853_D_ptr[4 * 15 + 2];
            temp_double_4 += K_extended_ptr[y_i * 3 + 2] * DOP853_D_ptr[4 * 15 + 3];


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
            const double delta_y = this->y_tmp[y_i] - this->y_old[y_i];
            const double sum_dy  = this->dy_tmp[y_i] + this->K_ptr[stride_K];
            Q_ptr[stride_Q + 4] = 2.0 * delta_y - this->step * sum_dy;

            // F[5] = h * f_old - delta_y
            Q_ptr[stride_Q + 5] = this->step * this->K_ptr[stride_K] - delta_y;

            // F[6] = delta_y
            Q_ptr[stride_Q + 6] = delta_y;

        }

        // Return values that were saved in temp variables back to state variables.
        this->load_back_from_temp();
        }
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

// ########################################################################################################################
// Explicit Runge - Kutta 2(3)
// ########################################################################################################################
void RK23::reset()
{

    // Allocate the size of K
    this->K.resize(this->num_y * 4);

    // Setup RK constants before calling the base class reset
    this->C_ptr     = RK23_C_ptr;
    this->A_ptr     = RK23_A_ptr;
    this->B_ptr     = RK23_B_ptr;
    this->E_ptr     = RK23_E_ptr;
    this->E3_ptr    = nullptr;       // Not used for RK23
    this->E5_ptr    = nullptr;       // Not used for RK23
    this->P_ptr     = RK23_P_ptr;       
    this->D_ptr     = nullptr;       // Not used for RK23
    this->K_ptr     = this->K.data();
    this->order     = RK23_order;
    this->n_stages  = RK23_n_stages;
    this->len_Acols = RK23_len_Acols;
    this->len_C     = RK23_len_C;
    this->len_Pcols = RK23_len_Pcols;
    this->error_estimator_order = RK23_error_estimator_order;
    this->error_exponent = RK23_error_exponent;
    this->integration_method = RK23_METHOD_INT;

    RKSolver::reset();
}


// ########################################################################################################################
// Explicit Runge - Kutta 4(5)
// ########################################################################################################################
void RK45::reset()
{
    // Allocate the size of K
    this->K.resize(this->num_y * 7);

    // Setup RK constants before calling the base class reset
    this->C_ptr     = RK45_C_ptr;
    this->A_ptr     = RK45_A_ptr;
    this->B_ptr     = RK45_B_ptr;
    this->E_ptr     = RK45_E_ptr;
    this->E3_ptr    = nullptr;       // Not used for RK45
    this->E5_ptr    = nullptr;       // Not used for RK45
    this->P_ptr     = RK45_P_ptr;
    this->D_ptr     = nullptr;       // Not used for RK45
    this->K_ptr     = this->K.data();
    this->order     = RK45_order;
    this->n_stages  = RK45_n_stages;
    this->len_Acols = RK45_len_Acols;
    this->len_C     = RK45_len_C;
    this->len_Pcols = RK45_len_Pcols;
    this->error_estimator_order = RK45_error_estimator_order;
    this->error_exponent = RK45_error_exponent;
    this->integration_method = RK45_METHOD_INT;

    RKSolver::reset();
}


// ########################################################################################################################
// Explicit Runge-Kutta Method of order 8(5,3) due Dormand & Prince
// ########################################################################################################################
void DOP853::reset()
{
    // Allocate the size of K
    this->K.resize(this->num_y * 18);
        // First 13 cols are K
        // Next 3 are K_extended
        // Next 2 are temp_double_array_ptr

    // Setup RK constants before calling the base class reset
    this->C_ptr     = DOP853_C_ptr;
    this->A_ptr     = DOP853_A_ptr;
    this->B_ptr     = DOP853_B_ptr;
    this->E_ptr     = nullptr;        // Not used for RK23
    this->E3_ptr    = DOP853_E3_ptr;
    this->E5_ptr    = DOP853_E5_ptr;
    this->P_ptr     = nullptr;        // TODO: Not implemented
    this->D_ptr     = nullptr;        // TODO: Not implemented
    this->K_ptr     = this->K.data();
    this->order     = DOP853_order;
    this->n_stages  = DOP853_n_stages;
    this->len_Acols = DOP853_A_cols;
    this->len_C     = DOP853_len_C;
    this->len_Pcols = DOP853_INTERPOLATOR_POWER; // Used by DOP853 dense output.
    this->error_estimator_order = DOP853_error_estimator_order;
    this->error_exponent = DOP853_error_exponent;
    this->integration_method = DOP853_METHOD_INT;

    RKSolver::reset();
}


void DOP853::p_estimate_error()
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

        // Dot product between K and E3 & E5 (sum over n_stages + 1; for DOP853 n_stages = 12
        // n = 0
        double temp_double = this->K_ptr[stride_K];
        double error_dot3  = this->E3_ptr[0] * temp_double;
        double error_dot5  = this->E5_ptr[0] * temp_double;

        // n = 1
        temp_double = this->K_ptr[stride_K + 1];
        error_dot3 += this->E3_ptr[1] * temp_double;
        error_dot5 += this->E5_ptr[1] * temp_double;

        // n = 2
        temp_double = this->K_ptr[stride_K + 2];
        error_dot3 += this->E3_ptr[2] * temp_double;
        error_dot5 += this->E5_ptr[2] * temp_double;

        // n = 3
        temp_double = this->K_ptr[stride_K + 3];
        error_dot3 += this->E3_ptr[3] * temp_double;
        error_dot5 += this->E5_ptr[3] * temp_double;

        // n = 4
        temp_double = this->K_ptr[stride_K + 4];
        error_dot3 += this->E3_ptr[4] * temp_double;
        error_dot5 += this->E5_ptr[4] * temp_double;

        // n = 5
        temp_double = this->K_ptr[stride_K + 5];
        error_dot3 += this->E3_ptr[5] * temp_double;
        error_dot5 += this->E5_ptr[5] * temp_double;

        // n = 6
        temp_double = this->K_ptr[stride_K + 6];
        error_dot3 += this->E3_ptr[6] * temp_double;
        error_dot5 += this->E5_ptr[6] * temp_double;

        // n = 7
        temp_double = this->K_ptr[stride_K + 7];
        error_dot3 += this->E3_ptr[7] * temp_double;
        error_dot5 += this->E5_ptr[7] * temp_double;

        // n = 8
        temp_double = this->K_ptr[stride_K + 8];
        error_dot3 += this->E3_ptr[8] * temp_double;
        error_dot5 += this->E5_ptr[8] * temp_double;

        // n = 9
        temp_double = this->K_ptr[stride_K + 9];
        error_dot3 += this->E3_ptr[9] * temp_double;
        error_dot5 += this->E5_ptr[9] * temp_double;

        // n = 10
        temp_double = this->K_ptr[stride_K + 10];
        error_dot3 += this->E3_ptr[10] * temp_double;
        error_dot5 += this->E5_ptr[10] * temp_double;

        // n = 11
        temp_double = this->K_ptr[stride_K + 11];
        error_dot3 += this->E3_ptr[11] * temp_double;
        error_dot5 += this->E5_ptr[11] * temp_double;

        // n = 12
        temp_double = this->K_ptr[stride_K + 12];
        error_dot3 += this->E3_ptr[12] * temp_double;
        error_dot5 += this->E5_ptr[12] * temp_double;

        // Find scale of y for error calculations
        const double scale_inv = 1.0 / (atol + std::fmax(std::fabs(this->y_old[y_i]), std::fabs(this->y_now[y_i])) * rtol);

        // We need the absolute value but since we are taking the square, it is guaranteed to be positive.
        // TODO: This will need to change if CySolver ever accepts complex numbers
        // error_norm_abs = fabs(error_dot_1)
        error_dot3 *= scale_inv;
        error_dot5 *= scale_inv;

        error_norm3 += (error_dot3 * error_dot3);
        error_norm5 += (error_dot5 * error_dot5);
    }

    // Check if errors are zero
    if ((error_norm5 == 0.0) && (error_norm3) == 0.0)
    {
        this->error_norm = 0.0;
    }
    else
    {
        double error_denom = error_norm5 + 0.01 * error_norm3;
        this->error_norm = this->step_size * error_norm5 / std::sqrt(error_denom * this->num_y_dbl);
    }
}
