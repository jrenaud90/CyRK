#include "cysolver.hpp"

// Constructors
CySolverBase::CySolverBase() {}
CySolverBase::CySolverBase(
    DiffeqFuncType diffeq_ptr,
    std::shared_ptr<CySolverResult> storage_ptr,
    const double t_start,
    const double t_end,
    const double* const y0_ptr,
    const unsigned int num_y,
    const unsigned int num_extra,
    const double* const args_ptr,
    const size_t max_num_steps,
    const size_t max_ram_MB) :
        status(0),
        num_y(num_y),
        num_extra(num_extra),
        t_start(t_start),
        t_end(t_end),
        storage_ptr(storage_ptr),
        diffeq_ptr(diffeq_ptr),
        args_ptr(args_ptr)
{   
    // Parse inputs
    this->capture_extra = num_extra > 0;
    
    // Setup storage
    this->storage_ptr->update_message("CySolverBase Initializing.");

    // Check for errors
    if (this->num_extra > (DY_LIMIT - Y_LIMIT))
    {
        this->status = -9;
        this->storage_ptr->error_code = -1;
        this->storage_ptr->update_message("CySolverBase Attribute Error: `num_extra` exceeds the maximum supported size.");
    }

    if (this->num_y > Y_LIMIT)
    {
        this->status = -9;
        this->storage_ptr->error_code = -1;
        this->storage_ptr->update_message("CySolverBase Attribute Error: `num_y` exceeds the maximum supported size.");
    }
    else if (this->num_y == 0)
    {
        this->status = -9;
        this->storage_ptr->error_code = -1;
        this->storage_ptr->update_message("CySolverBase Attribute Error: `num_y` = 0 so nothing to integrate.");
    }

    // Parse y values
    this->num_y_dbl  = (double)this->num_y;
    this->num_y_sqrt = std::sqrt(this->num_y_dbl);
    this->num_dy     = this->num_y + this->num_extra;
    // Make a copy of y0
    std::memcpy(this->y0_ptr, y0_ptr, sizeof(double) * this->num_y);

    // Parse time information
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
    MaxNumStepsOutput max_num_steps_output = find_max_num_steps(
        this->num_y,
        num_extra,
        max_num_steps,
        max_ram_MB
    );
    this->user_provided_max_num_steps = max_num_steps_output.user_provided_max_num_steps;
    this->max_num_steps = max_num_steps_output.max_num_steps;

    // Bind diffeq to C++ version
    this->diffeq = &CySolverBase::cy_diffeq;
}


// Destructors
CySolverBase::~CySolverBase()
{
    this->storage_ptr = nullptr;
    if (this->use_pysolver)
    {
        // Decrease reference count on the cython extension class instance
        Py_XDECREF(this->cython_extension_class_instance);
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
void CySolverBase::calc_first_step_size()
{
    // Overwritten by subclasses.
}

bool CySolverBase::check_status() const
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

void CySolverBase::cy_diffeq()
{
    // Call c function
    this->diffeq_ptr(this->dy_now_ptr, this->t_now_ptr[0], this->y_now_ptr, this->args_ptr);
}

void CySolverBase::reset()
{
    this->status = 0;
    this->reset_called = false;

    // Reset time
    this->t_now_ptr[0] = this->t_start;
    this->t_old = this->t_start;
    this->len_t = 1;

    // Reset ys
    std::memcpy(this->y_now_ptr, this->y0_ptr, sizeof(double) * this->num_y);
    std::memcpy(this->y_old_ptr, this->y0_ptr, sizeof(double) * this->num_y);

    // Call differential equation to set dy0
    this->diffeq(this);

    // Update dys
    std::memcpy(this->dy_old_ptr, this->dy_now_ptr, sizeof(double) * this->num_y);

    // Initialize storage
    this->storage_ptr->reset();
    this->storage_ptr->update_message("CySolverStorage reset, ready for data.");

    // Store initial conditions
    this->storage_ptr->save_data(this->t_now_ptr[0], this->y_now_ptr, this->dy_now_ptr);

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
        if (this->t_now_ptr[0] == this->t_end)
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
            this->storage_ptr->save_data(this->t_now_ptr[0], this->y_now_ptr, this->dy_now_ptr);
        }
    }

    // Note this is not an "else" block because the integrator may have finished with that last step.
    // Check status again to see if we are finished or there was an error in the last step
    if (this->status != 0)
    {
        // Update integration message
        this->storage_ptr->error_code = this->status;
        this->storage_ptr->success    = false;
        switch (this->status)
        {
        case 2:
            this->storage_ptr->update_message("Integration storage changed but integrator was not reset. Call `.reset()` before integrating after change.");
            break;
        case 1:
            this->storage_ptr->update_message("Integration completed without issue.");
            this->storage_ptr->success = true;
            break;
        case -1:
            this->storage_ptr->update_message("Error in step size calculation:\n\tRequired step size is less than spacing between numbers.");
            break;
        case -2:
            this->storage_ptr->update_message("Maximum number of steps (set by user) exceeded during integration.");
            break;
        case -3:
            this->storage_ptr->update_message("Maximum number of steps (set by system architecture) exceeded during integration.");
            break;
        case -4:
            this->storage_ptr->update_message("Error in step size calculation:\n\tError in step size acceptance.");
            break;
        case -9:
            this->storage_ptr->update_message("Error in CySolver initialization.");
            break;
        default:
            this->storage_ptr->update_message("Unknown status encountered during integration.");
            break;
        }
        
        // Call the finalizer on the storage class instance
        this->storage_ptr->finalize();
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


// Main Solve Method!
void CySolverBase::solve()
{
    while (this->check_status())
    {
        this->take_step();
    }
}


/* PySolver Methods */
// !!!
// Uncomment these dummy methods if working outside of CyRK and you just want the program to compile and run for testing/developing the C++ only code.
/*
bool import_CyRK__cy__pysolver_cyhook()
{
    return true;
}

int call_diffeq_from_cython(PyObject* x)
{
    return 1;
}

void Py_XINCREF(PyObject* x)
{
}
*/

void CySolverBase::set_cython_extension_instance(PyObject* cython_extension_class_instance, DiffeqMethod py_diffeq_method)
{
    this->use_pysolver = true;
    if (cython_extension_class_instance)
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
            this->storage_ptr->error_code = -51;
            this->storage_ptr->update_message("Error encountered importing python hooks.\n");
        }
        else
        {
            // TODO: Do we need to decref this at some point? During CySolver's deconstruction?
            Py_XINCREF(this->cython_extension_class_instance);
        }
    }
}

void CySolverBase::py_diffeq()
{
    // Call the differential equation in python space. Note that the optional arguments are handled by the python 
    // wrapper class. `this->args_ptr` is not used.
    call_diffeq_from_cython(this->cython_extension_class_instance, this->py_diffeq_method);
}
