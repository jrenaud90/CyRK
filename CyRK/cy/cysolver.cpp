#include "cysolver.hpp"
#include "dense.hpp"
#include "cysolution.hpp"

// !!!
// Uncomment these dummy methods if working outside of CyRK and you just want the program to compile and run for testing/developing the C++ only code.

/*
bool import_CyRK__cy__pysolver_cyhook()
{
    return true;
}

int call_diffeq_from_cython(PyObject* x, DiffeqMethod y)
{
    return 1;
}

void Py_XINCREF(PyObject* x)
{
}

void Py_XDECREF(PyObject* x)
{
}
*/

// Constructors
CySolverBase::CySolverBase() {}
CySolverBase::CySolverBase(
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
        PreEvalFunc pre_eval_func) :
            diffeq_ptr(diffeq_ptr),
            size_of_args(size_of_args),
            len_t_eval(len_t_eval),
            use_dense_output(use_dense_output),
            pre_eval_func(pre_eval_func),
            status(0),
            num_y(num_y),
            num_extra(num_extra),
            storage_sptr(storage_sptr),
            t_start(t_start),
            t_end(t_end)
{
    // Parse inputs
    this->capture_extra = num_extra > 0;

    // Setup storage
    this->storage_sptr->update_message("CySolverBase Initializing.");

    // Build storage for args
    if (args_ptr && (this->size_of_args > 0))
    {
        // Allocate memory for the size of args.
        // Store void pointer to it.
        this->args_char_vec.resize(this->size_of_args);

        // Copy over contents of arg
        this->args_ptr = this->args_char_vec.data();
        std::memcpy(this->args_ptr, args_ptr, this->size_of_args);
    }
    else
    {
        this->args_ptr = nullptr;
    }

    // Check for errors
    if (this->num_y == 0)
    {
        this->status = -9;
        this->storage_sptr->error_code = -1;
        this->storage_sptr->update_message("CySolverBase Attribute Error: `num_y` = 0 so nothing to integrate.");
    }

    // Parse y values
    this->num_y_dbl  = (double)this->num_y;
    this->num_y_sqrt = std::sqrt(this->num_y_dbl);
    this->num_dy     = this->num_y + this->num_extra;

    // Set up heap allocated arrays
    this->y0.resize(this->num_y);
    this->y_old.resize(this->num_y);
    this->y_now.resize(this->num_y);
    this->y_tmp.resize(this->num_y);
    this->y_interp.resize(this->num_y);
    // For dy, both the dy/dt and any extra outputs are stored. So the maximum size is `num_y` (25) + `num_extra` (25)
    this->dy_old.resize(this->num_dy);
    this->dy_now.resize(this->num_dy);
    this->dy_tmp.resize(this->num_dy);

    // Make a copy of y0
    std::memcpy(&this->y0[0], y0_ptr, sizeof(double) * this->num_y);
    
    // Parse t_eval
    if ((t_eval) && (this->len_t_eval > 0))
    {
        this->use_t_eval = true;
    }
    else
    {
        this->use_t_eval = false;
    }

    // Parse time information
    this->t_delta = t_end - t_start;
    this->t_delta_abs = std::fabs(this->t_delta);
    if (this->t_delta >= 0.0)
    {
        // Forward integration
        this->direction_flag = true;
        this->direction_inf = INF;

        if (this->use_t_eval)
        {
            // We do not need to copy the values since the integration is forward.
            this->t_eval_vec.resize(len_t_eval);
            this->t_eval_ptr = this->t_eval_vec.data();
            std::memcpy(this->t_eval_ptr, t_eval, sizeof(double) * len_t_eval);
        }
    }
    else {
        // Backward integration
        this->direction_flag = false;
        this->direction_inf = -INF;

        if (this->use_t_eval)
        {
            // We need to make sure that t_eval is properly sorted or the search algorithm will fail.
            // Need to make a copy because we do not want to change the array that was passed in by the user.
            this->t_eval_vec.resize(len_t_eval);
            this->t_eval_ptr = this->t_eval_vec.data();
            std::reverse_copy(t_eval, t_eval + len_t_eval, this->t_eval_ptr);
        }
    }

    // Parse maximum number of steps
    MaxNumStepsOutput max_num_steps_output = find_max_num_steps(
        this->num_y,
        num_extra,
        max_num_steps,
        max_ram_MB
    );
    this->user_provided_max_num_steps = max_num_steps_output.user_provided_max_num_steps;
    this->max_num_steps               = max_num_steps_output.max_num_steps;

    // Bind diffeq to C++ version
    this->diffeq = &CySolverBase::cy_diffeq;
}


// Destructors
CySolverBase::~CySolverBase()
{
    if (this->deconstruct_python)
    {
        // Decrease reference count on the cython extension class instance
        Py_XDECREF(this->cython_extension_class_instance);
    }

    // Reset shared pointers
    if (this->storage_sptr)
    {
        this->storage_sptr.reset();
    }
}

// Protected methods
void CySolverBase::p_estimate_error()
{
    // Overwritten by subclasses.
}

void CySolverBase::p_step_implementation()
{
    // Overwritten by subclasses.
}

// Public methods
void CySolverBase::offload_to_temp()
{
    /* Save "now" variables to temporary arrays so that the now array can be overwritten. */
    std::memcpy(&this->y_tmp[0], &this->y_now[0], sizeof(double) * this->num_y);
    std::memcpy(&this->dy_tmp[0], &this->dy_now[0], sizeof(double) * this->num_dy);
    this->t_tmp = this->t_now;
}

void CySolverBase::load_back_from_temp()
{
    /* Copy values from temporary array variables back into the "now" arrays. */
    std::memcpy(&this->y_now[0], &this->y_tmp[0], sizeof(double) * this->num_y);
    std::memcpy(&this->dy_now[0], &this->dy_tmp[0], sizeof(double) * this->num_dy);
    this->t_now = this->t_tmp;
}

void CySolverBase::set_Q_order(size_t* Q_order_ptr)
{
    // Overwritten by subclasses.
}

void CySolverBase::set_Q_array(double* Q_ptr)
{
    // Overwritten by subclasses.
}

void CySolverBase::calc_first_step_size()
{
    // Overwritten by subclasses.
}

NowStatePointers CySolverBase::get_now_state()
{
    return NowStatePointers(&this->t_now, this->y_now.data(), this->dy_now.data());
}

bool CySolverBase::check_status() const
{
    // If the solver is not in state 0 then that is an automatic rejection.
    if (this->status != 0)
    {
        return false;
    }

    // Otherwise, check if the solution storage is in an error state.
    if (this->storage_sptr) [[likely]]
    {
        if (this->storage_sptr->error_code != 0)
        {
            return false;
        }
    }
    else
    {
        // No storage!
        return false;
    }

    // If we reach here then we should be good to go.
    return true;
}

void CySolverBase::cy_diffeq() noexcept
{
    // Call c function
    this->diffeq_ptr(this->dy_now.data(), this->t_now, this->y_now.data(), this->args_ptr, this->pre_eval_func);
}

void CySolverBase::reset()
{
    this->status = 0;
    this->reset_called = false;

    // Reset runtime bools
    this->skip_t_eval = false;

    // Reset time
    this->t_now = this->t_start;
    this->t_old = this->t_start;
    this->len_t = 1;

    // Reset ys
    std::memcpy(&this->y_now[0], &this->y0[0], sizeof(double) * this->num_y);
    std::memcpy(&this->y_old[0], &this->y0[0], sizeof(double) * this->num_y);

    // Call differential equation to set dy0
    this->diffeq(this);

    // Update dys
    std::memcpy(&this->dy_old[0], &this->dy_now[0], sizeof(double) * this->num_y);

    // If t_eval is set then don't save initial conditions. They will be captured during stepping.
    if (!this->use_t_eval)
    {
        // Store initial conditions
        this->storage_sptr->save_data(this->t_now, &this->y_now[0], &this->dy_now[0]);
    }
    
    // Construct interpolator using t0 and y0 as its data point
    if (this->use_dense_output)
    {
        int built_dense = this->storage_sptr->build_dense(true);
        if (built_dense < 0)
        {
            this->status = -100;
        }
    }

    // Prep for t_eval
    if (this->direction_flag)
    {
        this->t_eval_index_old = 0;
    }
    else
    {
        this->t_eval_index_old = this->len_t_eval;
    }

    // Done with reset
    this->reset_called = true;
}

void CySolverBase::take_step()
{ 
    if (!this->reset_called) [[unlikely]]
    {
        // Reset must be called first.
        this->reset();
    }

    if (!this->status)
    {
        if (this->t_now == this->t_end) [[unlikely]]
        {
            // Integration finished
            this->t_old = this->t_end;
            this->status = 1;
        }
        else if (this->len_t >= this->max_num_steps) [[unlikely]]
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
        else [[likely]]
        {
            // ** Make call to solver's step implementation **
            bool save_data = true;
            bool prepare_for_next_step = true;

            this->p_step_implementation();
            this->len_t++;
            this->storage_sptr->steps_taken++;

            // Take care of dense output and t_eval
            int dense_built = 0;
            if (this->use_dense_output)
            {
                // We need to save many dense interpolators to storage. So let's heap allocate them (that is what "true" indicates)
                dense_built = this->storage_sptr->build_dense(true);
                if (dense_built < 0)
                {
                    this->status = -100;
                }
            }

            if (this->use_t_eval && !this->skip_t_eval)
            {
                // Don't save data at the end
                save_data = false;

                // Need to step through t_eval and call dense to determine correct data at each t_eval step.
                // Find the first index in t_eval that is close to current time.

                // Check if there are any t_eval steps between this new index and the last index.
                // Get lowest and highest indices
                auto lower_i = std::lower_bound(this->t_eval_vec.begin(), this->t_eval_vec.end(), this->t_now) - this->t_eval_vec.begin();
                auto upper_i = std::upper_bound(this->t_eval_vec.begin(), this->t_eval_vec.end(), this->t_now) - this->t_eval_vec.begin();
                
                size_t t_eval_index_new;
                if (lower_i == upper_i)
                {
                    // Only 1 index came back wrapping the value. See if it is different from before.
                    t_eval_index_new = lower_i;  // Doesn't matter which one we choose
                }
                else if (this->direction_flag)
                {
                    // Two different indicies returned. Since we are working our way from low to high values we want the upper one.
                    t_eval_index_new = upper_i;
                    if (t_eval_index_new == this->len_t_eval)
                    {
                        this->skip_t_eval = true;
                    }
                }
                else
                {
                    // Two different indicies returned. Since we are working our way from high to low values we want the lower one.
                    t_eval_index_new = lower_i;
                    if (t_eval_index_new == 0)
                    {
                        this->skip_t_eval = true;
                    }
                }

                int t_eval_index_delta;
                if (this->direction_flag)
                {
                    t_eval_index_delta = (int)t_eval_index_new - (int)this->t_eval_index_old;
                }
                else
                {
                    t_eval_index_delta = (int)this->t_eval_index_old - (int)t_eval_index_new;
                }
                
                // If t_eval_index_delta == 0 then there are no new interpolations required between the last integration step and now.
                // ^ In this case do not save any data, we are done with this step.
                if (t_eval_index_delta > 0)
                {
                    if (dense_built == 0)
                    {
                        // We are not saving interpolators to storage but we still need one to work on t_eval. 
                        // We will only ever need 1 interpolator per step. So let's just stack allocate that one.
                        dense_built = this->storage_sptr->build_dense(false);
                    }

                    // There are steps we need to interpolate over.
                    // Start with the old time and add t_eval step sizes until we are done.
                    // Create a y array and dy_array to use during interpolation

                    // If capture extra is set to true then we need to hold onto a copy of the current state
                    // The current state pointers must be overwritten if extra output is to be captured.
                    // However we need a copy of the current state pointers at the end of this step anyways. So just
                    // store them now and skip storing them later.

                    if (this->capture_extra)
                    {
                        // We need to copy the current state of y, dy, and time
                        this->t_old = this->t_now;
                        std::memcpy(&this->y_old[0], &this->y_now[0], sizeof(double) * this->num_y);
                        std::memcpy(&this->dy_old[0], &this->dy_now[0], sizeof(double) * this->num_dy);

                        // Don't update these again at the end
                        prepare_for_next_step = false;
                    }

                    for (int i = 0; i < t_eval_index_delta; i++)
                    {
                        double t_interp;
                        if (this->direction_flag)
                        {
                            t_interp = this->t_eval_ptr[this->t_eval_index_old + i];
                        }
                        else
                        {
                            t_interp = this->t_eval_ptr[this->t_eval_index_old - i - 1];
                        }

                        // Call the interpolator using this new time value.
                        this->storage_sptr->dense_vec.back().call(t_interp, this->y_interp.data());

                        if (this->capture_extra)
                        {
                            // If the user want to capture extra output then we also have to call the differential equation to get that extra output.
                            // To do this we need to hack the current integrators t_now, y_now, and dy_now.
                            // TODO: This could be more efficient if we just changed pointers but since the PySolver only stores y_now_ptr, dy_now_ptr, etc at initialization, it won't be able to see changes to new pointer. 
                            // So for now we have to do a lot of copying of data.

                            // Copy the interpreted y onto the current y_now_ptr. Also update t_now
                            this->t_now = t_interp;
                            std::memcpy(&this->y_now[0], this->y_interp.data(), sizeof(double) * this->num_y);

                            // Call diffeq to update dy_now_ptr with the extra output.
                            this->diffeq(this);
                        }
                        // Save interpolated data to storage. If capture extra is true then dy_now holds those extra values. If it is false then it won't hurt to pass dy_now to storage.
                        this->storage_sptr->save_data(t_interp, this->y_interp.data(), &this->dy_now[0]);
                    }
                }
                // Update the old index for the next step
                this->t_eval_index_old = t_eval_index_new;
            }
            if (save_data)
            {
                // No data has been saved from the current step. Save the integrator data for this step as the solution.
                this->storage_sptr->save_data(this->t_now, &this->y_now[0], &this->dy_now[0]);
            }

            if (prepare_for_next_step)
            {
                // Prep for next step
                this->t_old = this->t_now;
                std::memcpy(&this->y_old[0], &this->y_now[0], sizeof(double) * this->num_y);
                std::memcpy(&this->dy_old[0], &this->dy_now[0], sizeof(double) * this->num_dy);
            }
        }
    }

    // Note this is not an "else" block because the integrator may have finished with that last step.
    // Check status again to see if we are finished or there was an error in the last step
    if (this->status != 0)
    {
        // Update integration message
        this->storage_sptr->error_code = this->status;
        this->storage_sptr->success    = false;
        switch (this->status)
        {
        case 2:
            this->storage_sptr->update_message("Integration storage changed but integrator was not reset. Call `.reset()` before integrating after change.");
            break;
        case 1:
            this->storage_sptr->update_message("Integration completed without issue.");
            this->storage_sptr->success = true;
            break;
        case -1:
            this->storage_sptr->update_message("Error in step size calculation:\n\tRequired step size is less than spacing between numbers.");
            break;
        case -2:
            this->storage_sptr->update_message("Maximum number of steps (set by user) exceeded during integration.");
            break;
        case -3:
            this->storage_sptr->update_message("Maximum number of steps (set by system architecture) exceeded during integration.");
            break;
        case -4:
            this->storage_sptr->update_message("Error in step size calculation:\n\tError in step size acceptance.");
            break;
        case -9:
            this->storage_sptr->update_message("Error in CySolver initialization.");
            break;
        default:
            this->storage_sptr->update_message("Unknown status encountered during integration.");
            break;
        }
    }
}


// Main Solve Method!
void CySolverBase::solve()
{
    while (this->check_status())
    {
        this->take_step();
    }
}


/* PySolver Methods */
void CySolverBase::set_cython_extension_instance(PyObject* cython_extension_class_instance, DiffeqMethod py_diffeq_method)
{
    this->use_pysolver = true;
    if (cython_extension_class_instance) [[likely]]
    {
        this->cython_extension_class_instance = cython_extension_class_instance;
        this->py_diffeq_method = py_diffeq_method;

        // Change diffeq binding to the python version
        this->diffeq = &CySolverBase::py_diffeq;

        // Import the cython/python module (functionality provided by "pysolver_api.h")
        const int import_error = import_CyRK__cy__pysolver_cyhook();
        if (import_error)
        {
            this->use_pysolver = false;
            this->status = -1;
            this->storage_sptr->error_code = -51;
            this->storage_sptr->update_message("Error encountered importing python hooks.\n");
        }
        else
        {
            Py_XINCREF(this->cython_extension_class_instance);
            this->deconstruct_python = true;
        }
    }
}

void CySolverBase::py_diffeq()
{
    // Call the differential equation in python space. Note that the optional arguments are handled by the python 
    // wrapper class. `this->args_ptr` is not used.
    call_diffeq_from_cython(this->cython_extension_class_instance, this->py_diffeq_method);
}
