#include "cysolver.hpp"

// Constructors
CySolverBase::CySolverBase() {}
CySolverBase::CySolverBase(
        DiffeqFuncType diffeq_ptr,
        std::shared_ptr<CySolverResult> storage_ptr,
        const double t_start,
        const double t_end,
        double* y0_ptr,
        size_t num_y,
        bool capture_extra,
        size_t num_extra,
        double* args_ptr,
        size_t max_num_steps,
        size_t max_ram_MB)
{   
    // Assume no errors for now.
    this->reset_called = false;
    this->status = 0;

    // Setup storage
    this->storage_ptr = storage_ptr;

    this->storage_ptr->update_message("CySolverBase Initializing.");

    // Check for errors
    if (capture_extra)
    {
        if (num_extra == 0)
        {
            this->storage_ptr->error_code = -1;
            this->storage_ptr->update_message("CySolverBase Attribute Error: `capture_extra` set to True, but `num_extra` set to 0.");
        }
        else if (num_extra > 25)
        {
            this->storage_ptr->error_code = -1;
            this->storage_ptr->update_message("CySolverBase Attribute Error: `num_extra` exceeds the maximum supported value of 25.");
        }

    }
    else if (num_extra > 0)
    {
        this->storage_ptr->error_code = -1;
        this->storage_ptr->update_message("CySolverBase Attribute Error: `capture_extra` set to False, but `num_extra` > 0.");
    }

    if (num_y > 25)
    {
        this->storage_ptr->error_code = -1;
        this->storage_ptr->update_message("CySolverBase Attribute Error: `num_y` exceeds the maximum supported value of 25.");
    }
    else if (num_y == 0)
    {
        this->storage_ptr->error_code = -1;
        this->storage_ptr->update_message("CySolverBase Attribute Error: Integration completed. `num_y` = 0 so nothing to integrate.");
    }

    // Parse differential equation
    this->diffeq_ptr = diffeq_ptr;
    this->args_ptr = args_ptr;

    // Parse capture extra information
    this->capture_extra = capture_extra;
    this->num_extra = num_extra;

    // Parse y values
    this->num_y      = num_y;
    this->num_y_dbl  = (double)this->num_y;
    this->num_y_sqrt = std::sqrt(this->num_y_dbl);
    this->num_dy     = this->num_y + this->num_extra;
    // Make a copy of y0
    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        this->y0_ptr[y_i] = y0_ptr[y_i];
    }

    // Parse time information
    this->t_start = t_start;
    this->t_end = t_end;
    this->t_delta = t_end - t_start;
    this->t_delta_abs = std::fabs(this->t_delta);
    if (this->t_delta >= 0.0)
    {
        // Forward integration
        this->direction_flag = true;
        this->direction_inf = INF;
    }
    else {
        // Backward integration
        this->direction_flag = false;
        this->direction_inf = -INF;
    }

    // Parse maximum number of steps
    find_max_num_steps(
        this->num_y,
        num_extra,
        max_num_steps,
        max_ram_MB,
        capture_extra,
        &this->user_provided_max_num_steps,
        &this->max_num_steps);
}


// Destructors
CySolverBase::~CySolverBase()
{
    this->storage_ptr = nullptr;
}


// Protected methods
void CySolverBase::p_step_implementation()
{
    // Overwritten by subclasses.
}


// Public methods
bool CySolverBase::check_status()
{
    // If the solver is not in state 0 then that is an automatic rejection.
    if (this->status != 0)
    {
        return false;
    }

    // Otherwise, check if the solution storage is in an error state.
    if (this->storage_ptr)
    {
        if (this->storage_ptr->error_code != 0)
        {
            return false;
        }
    }

    // If we reach here then we should be good to go.
    return true;
}

void CySolverBase::diffeq()
{
    // Call the user-provided differential equation
    this->diffeq_ptr(this->dy_now_ptr, this->t_now, this->y_now_ptr, this->args_ptr);
}

void CySolverBase::reset()
{
    double temp_double;

    this->status = 0;
    this->reset_called = false;

    // Reset time
    this->t_now = this->t_start;
    this->t_old = this->t_start;
    this->len_t = 1;

    // Reset ys
    for (size_t y_i = 0; y_i < this->num_y; y_i++)
    {
        temp_double = this->y0_ptr[y_i];
        this->y_now_ptr[y_i] = temp_double;
        this->y_old_ptr[y_i] = temp_double;
    }

    // Call differential equation to set dy0
    this->diffeq();

    // Update dys
    for (size_t dy_i = 0; dy_i < this->num_dy; dy_i++)
    {
        this->dy_old_ptr[dy_i] = this->dy_now_ptr[dy_i];
    }

    // Inititialize storage
    this->storage_ptr->reset();
    this->storage_ptr->update_message("CySolverStorage reset, ready for data.");

    // Store initial conditions
    this->storage_ptr->save_data(this->t_now, this->y_now_ptr, this->dy_now_ptr);


    this->reset_called = true;
}

void CySolverBase::take_step()
{
    if (!this->reset_called)
    {
        // Reset must be called first.
        this->reset();
    }

    if (this->status == 0)
    {
        if (this->t_now == this->t_end)
        {
            // Integration finished
            this->t_old = this->t_end;
            this->status = 1;
        }
        else if (this->len_t >= this->max_num_steps)
        {
            if (this->user_provided_max_num_steps)
            {
                // Maximum number of steps reached (as set by user).
                this->status = -2;
            }
            else {
                // Maximum number of steps reached (as set by RAM limitations).
                this->status = -3;
            }
        }
        else
        {
            // ** Make call to solver's step implementation **
            this->p_step_implementation();
            this->len_t++;

            // Save data
            this->storage_ptr->save_data(this->t_now, this->y_now_ptr, this->dy_now_ptr);
        }
    }

    // Check status again to see if we are finished or there was an error in the last step
    if (this->status != 0)
    {
        // Update integration message
        this->storage_ptr->error_code = this->status;
        switch (this->status)
        {
        case 2:
            this->storage_ptr->update_message("Integration storage was changed but integrator was not reset. Call `.reset()` before integrating after storage change.\n");
            break;
        case 1:
            this->storage_ptr->update_message("Integration completed without issue.\n");
            break;
        case -1:
            this->storage_ptr->update_message("Error in step size calculation:\n\tRequired step size is less than spacing between numbers.\n");
            break;
        case -2:
            this->storage_ptr->update_message("Maximum number of steps (set by user) exceeded during integration.\n");
            break;
        case -3:
            this->storage_ptr->update_message("Maximum number of steps (set by system architecture) exceeded during integration.\n");
            break;
        case -4:
            this->storage_ptr->update_message("Error in step size calculation:\n\tError in step size acceptance.\n");
            break;
        default:
            this->storage_ptr->update_message("Unknown status encountered during integration.\n");
            break;
        }
    }
}


void CySolverBase::change_storage(std::shared_ptr<CySolverResult> new_storage_ptr, bool auto_reset)
{
    // Change the storage reference. 
    this->storage_ptr = new_storage_ptr;
    
    // Set status to indicate that storage has changed but reset has not been updated
    this->status = 2;

    // Changing storage requires a reset
    if (auto_reset)
    {
        this->reset();
    }
}
