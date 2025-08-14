#include <stdexcept>
#include <numeric>

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

/* ========================================================================= */
/* ========================  Configurations  =============================== */
/* ========================================================================= */
ProblemConfig::ProblemConfig()
{
}

ProblemConfig::ProblemConfig(
        DiffeqFuncType diffeq_ptr_,
        double t_start_,
        double t_end_,
        std::vector<double>& y0_vec_): 
            diffeq_ptr(diffeq_ptr_),
            t_start(t_start_),
            t_end(t_end_),
            y0_vec(y0_vec_)
{
    this->initialize();
}

ProblemConfig::ProblemConfig(
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
        bool force_retain_solver_): 
            diffeq_ptr(diffeq_ptr_),
            t_start(t_start_),
            t_end(t_end_),
            y0_vec(y0_vec_),
            args_vec(args_vec_),
            t_eval_vec(t_eval_vec_),
            num_extra(num_extra_),
            expected_size(expected_size_),
            max_num_steps(max_num_steps_),
            max_ram_MB(max_ram_MB_),
            pre_eval_func(pre_eval_func_),
            capture_dense_output(capture_dense_output_),
            force_retain_solver(force_retain_solver_)
{
    this->initialize();
}

void ProblemConfig::update_properties(
        DiffeqFuncType diffeq_ptr_,
        double t_start_,
        double t_end_,
        std::vector<double>& y0_vec_)
{
    this->diffeq_ptr = diffeq_ptr_;
    this->t_start    = t_start_;
    this->t_end      = t_end_;
    this->y0_vec     = y0_vec_;

    this->initialize();
}

void ProblemConfig::update_properties(
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
        bool force_retain_solver_)
{
    this->diffeq_ptr    = diffeq_ptr_;
    this->t_start       = t_start_;
    this->t_end         = t_end_;
    this->y0_vec        = y0_vec_;
    this->args_vec      = args_vec_;
    this->t_eval_vec    = t_eval_vec_;
    this->num_extra     = num_extra_;
    this->expected_size = expected_size_;
    this->max_num_steps = max_num_steps_;
    this->max_ram_MB    = max_ram_MB_;
    this->pre_eval_func = pre_eval_func_;
    this->capture_dense_output = capture_dense_output_;
    this->force_retain_solver  = force_retain_solver_;

    this->initialize();
}

void ProblemConfig::initialize()
{
    this->initialized = false;
    if (this->y0_vec.size() == 0) [[unlikely]]
    {
        throw std::length_error("Unexpected size of y0_vec; at least one dependent variable is required.");
    }

    round_to_2(this->expected_size);
    this->capture_extra   = this->num_extra > 0;
    this->t_eval_provided = this->t_eval_vec.size() > 0;
    this->num_y           = this->y0_vec.size();
    this->num_dy          = this->num_y + this->num_extra;
    this->num_y_dbl       = (double)this->num_y;
    this->num_y_sqrt      = std::sqrt(this->num_y_dbl);
    this->num_dy_dbl      = (double)this->num_dy;
    this->initialized     = true;
}

void ProblemConfig::update_properties_from_config(ProblemConfig* new_config_ptr)
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
        new_config_ptr->force_retain_solver
    );
}

/* ========================================================================= */
/* =========================  Constructors  ================================ */
/* ========================================================================= */
CySolverBase::CySolverBase() :
        integration_method(ODEMethod::BASE_METHOD)
{

}

CySolverBase::CySolverBase(CySolverResult* storage_ptr_) : 
        storage_ptr(storage_ptr_),
        integration_method(ODEMethod::BASE_METHOD)
{
    // Base constructor does not do much.
}

/* ========================================================================= */
/* =========================  Deconstructors  ============================== */
/* ========================================================================= */
CySolverBase::~CySolverBase()
{
    // Deconstruct python-related properties
    this->clear_python_refs();

    // Clear vectors
    this->t_eval_reverse_vec.clear();
    this->y_holder_vec.clear();
    this->dy_holder_vec.clear();
}

/* ========================================================================= */
/* =========================  Protected Methods  =========================== */
/* ========================================================================= */
CyrkErrorCodes CySolverBase::p_additional_setup() noexcept
{
    // Overwritten by subclasses.
    return CyrkErrorCodes::NO_ERROR;
}

void CySolverBase::p_estimate_error() noexcept
{
    // Overwritten by subclasses.
}

void CySolverBase::p_step_implementation() noexcept
{
    // Overwritten by subclasses.
}

inline void CySolverBase::p_cy_diffeq() noexcept
{
    // Call c function
    this->diffeq_ptr(
        this->dy_now_ptr,
        this->t_now,
        this->y_now_ptr,
        this->args_ptr,
        this->pre_eval_func);
}

void CySolverBase::p_calc_first_step_size() noexcept
{
    // Overwritten by subclasses.
}

/* ========================================================================= */
/* =========================  Public Methods  ============================== */
/* ========================================================================= */
void CySolverBase::set_Q_order(size_t* Q_order_ptr)
{
    // Overwritten by subclasses.
}

void CySolverBase::set_Q_array(double* Q_ptr)
{
    // Overwritten by subclasses.
}

void CySolverBase::clear_python_refs()
{
    if (this->cython_extension_class_instance)
    {
        this->cython_extension_class_instance = nullptr;
        this->use_pysolver                    = false;
    }
}

void CySolverBase::offload_to_temp() noexcept
{
    /* Save "now" variables to temporary arrays so that the now array can be overwritten. */
    std::memcpy(this->y_tmp_ptr, this->y_now_ptr, this->sizeof_dbl_Ny);
    std::memcpy(this->dy_tmp_ptr, this->dy_now_ptr, this->sizeof_dbl_Ndy);
    this->t_tmp = this->t_now;
}

void CySolverBase::load_back_from_temp() noexcept
{
    /* Copy values from temporary array variables back into the "now" arrays. */
    std::memcpy(this->y_now_ptr, this->y_tmp_ptr, this->sizeof_dbl_Ny);
    std::memcpy(this->dy_now_ptr, this->dy_tmp_ptr, this->sizeof_dbl_Ndy);
    this->t_now = this->t_tmp;
}

CyrkErrorCodes CySolverBase::resize_num_y(size_t num_y_, size_t num_dy_)
{    
    // Setup y-vectors and pointers
    try
    {
        this->y_holder_vec.resize(num_y_ * 4); // 4 is the number of subarrays held in this vector.
        this->dy_holder_vec.resize(num_dy_ * 3); // 3 is the number of subarrays held in this vector.
    }
    catch (const std::bad_alloc&)
    {
        return CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }
    double* y_holder_ptr = this->y_holder_vec.data();
    this->y_old_ptr      = &y_holder_ptr[0];
    this->y_now_ptr      = &y_holder_ptr[num_y_];
    this->y_tmp_ptr      = &y_holder_ptr[num_y_ * 2];
    this->y_interp_ptr   = &y_holder_ptr[num_y_ * 3];
    // Repeat for dy; dy holds num_y + num_extra values.
    double* dy_holder_ptr = this->dy_holder_vec.data();
    this->dy_old_ptr      = &dy_holder_ptr[0];
    this->dy_now_ptr      = &dy_holder_ptr[num_dy_];
    this->dy_tmp_ptr      = &dy_holder_ptr[num_dy_ * 2];

    return CyrkErrorCodes::NO_ERROR;
}

CyrkErrorCodes CySolverBase::setup()
{
    CyrkErrorCodes setup_status = CyrkErrorCodes::NO_ERROR;

    // Reset flags
    this->t_eval_finished = false;
    this->setup_called    = false;
    this->error_flag      = false;
    this->user_provided_max_num_steps = false;
    this->clear_python_refs();

    while (setup_status == CyrkErrorCodes::NO_ERROR)
    {
        // Check that everything has been initialized properly.
        if (not this->storage_ptr)
        {
            setup_status = CyrkErrorCodes::PROPERTY_NOT_SET;
            break;
        }
        if (not this->storage_ptr->config_uptr)
        {
            setup_status = CyrkErrorCodes::PROPERTY_NOT_SET;
            break;
        }
        
        // Setup PySolver
        if (this->storage_ptr->config_uptr->cython_extension_class_instance and 
            this->storage_ptr->config_uptr->py_diffeq_method)
        {
            this->set_cython_extension_instance(
                this->storage_ptr->config_uptr->cython_extension_class_instance,
                this->storage_ptr->config_uptr->py_diffeq_method
            );
        }

        // For performance reasons we will store a few parameters from the config into this object. 
        this->num_y            = this->storage_ptr->config_uptr->num_y;
        this->num_extra        = this->storage_ptr->config_uptr->num_extra;
        this->num_dy           = this->storage_ptr->config_uptr->num_dy;
        this->sizeof_dbl_Ny    = sizeof(double) * this->num_y;
        this->sizeof_dbl_Ndy   = sizeof(double) * this->num_dy;
        this->num_y_dbl        = this->storage_ptr->config_uptr->num_y_dbl;
        this->num_y_sqrt       = this->storage_ptr->config_uptr->num_y_sqrt;
        this->capture_extra    = this->num_extra > 0;
        this->use_dense_output = this->storage_ptr->config_uptr->capture_dense_output;

        // Setup time information
        this->len_t          = 0;
        this->t_start        = this->storage_ptr->config_uptr->t_start;
        this->t_end          = this->storage_ptr->config_uptr->t_end;
        this->t_delta        = this->t_end - this->t_start;
        this->t_delta_abs    = std::fabs(this->t_delta);
        this->direction_flag = this->t_delta >= 0.0;
        this->direction_inf  = (this->direction_flag) ? INF : -INF;

        // Pull out pointers to other data storage.
        this->diffeq_ptr    = this->storage_ptr->config_uptr->diffeq_ptr;
        this->pre_eval_func = this->storage_ptr->config_uptr->pre_eval_func;
        this->size_of_args  = this->storage_ptr->config_uptr->args_vec.size();
        this->args_ptr      = this->storage_ptr->config_uptr->args_vec.data();
        this->len_t_eval    = this->storage_ptr->config_uptr->t_eval_vec.size();
        this->use_t_eval    = this->len_t_eval > 0;
        this->t_eval_ptr    = this->storage_ptr->config_uptr->t_eval_vec.data();

        // Setup y-vectors and pointers
        this->y0_ptr = this->storage_ptr->config_uptr->y0_vec.data();
    
        // Resize the vectors now that we know the number of ys and dys.
        setup_status = this->resize_num_y(this->num_y, this->num_dy);

        // Handle backward integration.
        if (not this->direction_flag)
        {
            if (this->use_t_eval)
            {
                // We need to make sure that t_eval is properly sorted or the search algorithm will fail.
                // Need to make a copy because we do not want to change the array that was passed in by the user.
                try
                {
                    this->t_eval_reverse_vec.resize(len_t_eval);
                }
                catch (const std::bad_alloc&)
                {
                    setup_status = CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
                    break;
                }
                std::reverse_copy(this->t_eval_ptr, this->t_eval_ptr + len_t_eval, this->t_eval_reverse_vec.data());
                // Change the t_eval pointer to this new vector.
                this->t_eval_ptr = this->t_eval_reverse_vec.data();
            }
        }

        // Parse maximum number of steps
        MaxNumStepsOutput max_num_steps_output = find_max_num_steps(
            this->num_y,
            this->num_extra,
            this->storage_ptr->config_uptr->max_num_steps,
            this->storage_ptr->config_uptr->max_ram_MB
        );
        this->user_provided_max_num_steps = max_num_steps_output.user_provided_max_num_steps;
        this->max_num_steps = max_num_steps_output.max_num_steps;

        if (this->use_pysolver)
        {
            // Change diffeq binding to the python version
            this->diffeq = &CySolverBase::py_diffeq;
        }
        else
        {
            // Bind diffeq to C++ version
            this->diffeq = &CySolverBase::p_cy_diffeq;
        }

        // Some methods require additional setup before the current state is set.
        setup_status = this->p_additional_setup();
        if (setup_status != CyrkErrorCodes::NO_ERROR)
        {
            break;
        }

        // Reset state to start; run diffeq to get that first dy/dt value.
        this->t_tmp = 0.0;
        this->t_now = this->t_start;
        this->t_old = this->t_start;
        this->len_t = 0;

        // Reset ys
        std::memcpy(this->y_old_ptr, this->y0_ptr, this->sizeof_dbl_Ny);
        std::memcpy(this->y_now_ptr, this->y0_ptr, this->sizeof_dbl_Ny);

        // Call differential equation to set dy0
        if ((not this->diffeq_ptr) and (not this->py_diffeq_method))
        {
            // If the user did not provide a diffeq then we cannot continue.
            setup_status = CyrkErrorCodes::PROPERTY_NOT_SET;
            break;
        }
        this->diffeq(this);

        // Update dys
        std::memcpy(this->dy_old_ptr, this->dy_now_ptr, this->sizeof_dbl_Ndy);

        // If t_eval is set then don't save initial conditions. They will be captured during stepping.
        if (not this->use_t_eval)
        {
            // Store initial conditions
            this->storage_ptr->save_data(this->t_now, this->y_now_ptr, this->dy_now_ptr);
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

        // Construct interpolator using t0 and y0 as its data point
        if (this->use_dense_output and (setup_status == CyrkErrorCodes::NO_ERROR))
        {
            this->storage_ptr->build_dense(true);
        }

        break;
    }
    // Done with setup
    if (setup_status == CyrkErrorCodes::NO_ERROR)
    {
        this->setup_called = true;
    }
    return setup_status;
}

NowStatePointers CySolverBase::get_now_state()
{
    return NowStatePointers(&this->t_now, this->y_now_ptr, this->dy_now_ptr);
}

inline bool CySolverBase::check_status() const
{
    if (this->storage_ptr) [[likely]]
    {
        // We want to return false for any non-error status, even successful integration.
        return (this->storage_ptr->status == CyrkErrorCodes::NO_ERROR) and (not this->error_flag) and this->setup_called;
    }
    return false;
}

void CySolverBase::take_step()
{ 
    // We assume `this->check_status()` is true before this method was called.
    // Don't need to check it again.
    if (this->t_now == this->t_end) [[unlikely]]
    {
        // Integration finished we will still coninue with this last time step.
        this->t_old = this->t_end;
        this->storage_ptr->update_status(CyrkErrorCodes::SUCCESSFUL_INTEGRATION);
    }
    else if (this->len_t >= this->max_num_steps) [[unlikely]]
    {
        if (this->user_provided_max_num_steps)
        {
            // Maximum number of steps reached (as set by user).
            this->error_flag = true;
            this->storage_ptr->update_status(CyrkErrorCodes::MAX_STEPS_USER_EXCEEDED);
        }
        else
        {
            // Maximum number of steps reached (as set by RAM limitations).
            this->error_flag = true;
            this->storage_ptr->update_status(CyrkErrorCodes::MAX_STEPS_SYSARCH_EXCEEDED);
        }
    }
    else [[likely]]
    {
        // ** Make call to solver's step implementation **
        bool save_data              = true;
        bool prepare_for_next_step  = true;
        bool dense_built            = false;

        this->p_step_implementation();
        this->len_t++;
        this->storage_ptr->steps_taken++;

        // Take care of dense output and t_eval
        if (this->use_dense_output)
        {
            // We need to save many dense interpolators to storage. So let's heap allocate them (that is what "true" indicates)
            this->storage_ptr->build_dense(true);
            dense_built = true;
        }

        // Check if we are saving data at intermediate steps pulled from t_eval.
        if (this->use_t_eval and (not this->t_eval_finished) and (not this->error_flag))
        {
            // Don't save data at the end since we will save it during the interpolation steps.
            save_data = false;

            // Need to step through t_eval and call dense to determine correct data at each t_eval step.
            // Find the first index in t_eval that is close to current time.

            // Check if there are any t_eval steps between this new index and the last index.
            // Get lowest and highest indices
            auto lower_i = std::lower_bound(this->t_eval_ptr, this->t_eval_ptr + this->len_t_eval, this->t_now) - this->t_eval_ptr;
            auto upper_i = std::upper_bound(this->t_eval_ptr, this->t_eval_ptr + this->len_t_eval, this->t_now) - this->t_eval_ptr;
                
            size_t t_eval_index_new;
            if (lower_i == upper_i)
            {
                // Only 1 index came back wrapping the value. See if it is different from before.
                t_eval_index_new = lower_i;  // Doesn't matter which one we choose
            }
            else if (this->direction_flag)
            {
                // 2+ indices returned.
                // For forward integration (direction_flag), we are working our way from low to high values we want the upper one.
                t_eval_index_new = upper_i;
                if (t_eval_index_new == this->len_t_eval)
                {
                    // We are at the boundary of the t_eval array. Don't try to record t_eval for the rest of the integration.
                    this->t_eval_finished = true;
                }
            }
            else
            {
                // 2+ indices returned.
                // For backward integration (direction_flag), we are working our way from high to low values we want the lower one.
                t_eval_index_new = lower_i;
                if (t_eval_index_new == 0)
                {
                    // We are at the boundary of the t_eval array. Don't try to record t_eval for the rest of the integration.
                    this->t_eval_finished = true;
                }
            }

            size_t t_eval_index_delta = 0;
            bool t_eval_grt_zero = false;
            if (this->direction_flag)
            {
                t_eval_grt_zero    = (t_eval_index_new > this->t_eval_index_old);
                t_eval_index_delta = t_eval_index_new - this->t_eval_index_old;
            }
            else
            {
                t_eval_grt_zero    = (this->t_eval_index_old > t_eval_index_new);
                t_eval_index_delta = this->t_eval_index_old - t_eval_index_new;
            }
                
            // If t_eval_index_delta == 0 then there are no new interpolations required between the last integration step and now.
            // ^ In this case do not save any data, we are done with this step.
            if (t_eval_grt_zero)
            {
                if (not dense_built)
                {
                    // We are not saving interpolators to storage but we still need one to work on t_eval. 
                    // We will only ever need 1 interpolator per step.
                    // The `false` flag tells the storage to not to append the interpolator. 
                    // One interpolated will be overwritten at each step.
                    this->storage_ptr->build_dense(false);
                    dense_built = true;
                }

                // There are steps we need to interpolate over.
                // Start with the old time and add t_eval step sizes until we are done.
                // Create a y-array and dy-array to use during interpolation.

                // If capture extra is set to true then we need to hold onto a copy of the current state
                // The current state pointers must be overwritten if extra output is to be captured.
                // However we need a copy of the current state pointers at the end of this step anyways. So just
                // store them now and skip storing them later.
                if (this->capture_extra and (not this->error_flag))
                {
                    // We need to copy the current state of y, dy, and time
                    this->t_old = this->t_now;
                    std::memcpy(this->y_old_ptr, this->y_now_ptr, this->sizeof_dbl_Ny);
                    std::memcpy(this->dy_old_ptr, this->dy_now_ptr, this->sizeof_dbl_Ndy);

                    // Don't update these again at the end
                    prepare_for_next_step = false;
                }

                if ((not this->error_flag) and dense_built) [[likely]]
                {
                    for (size_t i = 0; i < t_eval_index_delta; i++)
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
                        // Call the last dense solution saved. If we are continuously saving them then this will be the last one.
                        // If we are not saving them then this will be the first and last one (only ever is length 1).
                        this->storage_ptr->dense_vec.back().call(t_interp, this->y_interp_ptr);

                        if (this->capture_extra)
                        {
                            // If the user want to capture extra output then we also have to call the differential equation to get that extra output.
                            // To do this we need to hack the current integrators t_now, y_now, and dy_now.
  
                            // TODO: This could be more efficient if we just changed pointers but since the PySolver only stores y_now_ptr, dy_now_ptr, etc at initialization, it won't be able to see changes to new pointer. 
                            // So for now we have to do a lot of copying of data.

                            // Copy the interpreted y onto the current y_now_ptr. Also update t_now
                            this->t_now = t_interp;
                            std::memcpy(this->y_now_ptr, this->y_interp_ptr, this->sizeof_dbl_Ny);

                            // Call diffeq to update dy_now_ptr with the extra output.
                            this->diffeq(this);
                        }
                        // Save interpolated data to storage. If capture extra is true then dy_now holds those extra values. If it is false then it won't hurt to pass dy_now to storage.
                        this->storage_ptr->save_data(t_interp, this->y_interp_ptr, this->dy_now_ptr);
                    }
                }
            }
            // Update the old index for the next step
            this->t_eval_index_old = t_eval_index_new;
        }
        if (save_data and (not this->error_flag))
        {
            // No data has been saved from the current step. Save the integrator data for this step as the solution.
            this->storage_ptr->save_data(this->t_now, this->y_now_ptr, this->dy_now_ptr);
        }

        if (prepare_for_next_step and (not this->error_flag))
        {
            // Prep for next step
            this->t_old = this->t_now;
            std::memcpy(this->y_old_ptr, this->y_now_ptr, this->sizeof_dbl_Ny);
            std::memcpy(this->dy_old_ptr, this->dy_now_ptr, this->sizeof_dbl_Ndy);
        }
    }

    // Check if the integration is finished and successful.
    if (this->storage_ptr->status == CyrkErrorCodes::SUCCESSFUL_INTEGRATION)
    {
        this->storage_ptr->success = true;
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


/* ========================================================================= */
/* ========================  PySolver Methods  ============================= */
/* ========================================================================= */
CyrkErrorCodes CySolverBase::set_cython_extension_instance(
        PyObject* cython_extension_class_instance,
        DiffeqMethod py_diffeq_method)
{
    // First check to see if a python instance has already been installed in this function
    // i.e., setup is being called multiple times.
    this->clear_python_refs();

    // Now proceed to installing python functions.
    this->use_pysolver = true;
    if (cython_extension_class_instance) [[likely]]
    {
        this->cython_extension_class_instance = cython_extension_class_instance;
        this->py_diffeq_method                = py_diffeq_method;

        // Import the cython/python module (functionality provided by "pysolver_api.h")
        const int import_error = import_CyRK__cy__pysolver_cyhook();
        if (import_error) [[unlikely]]
        {
            this->use_pysolver = false;
            this->storage_ptr->update_status(CyrkErrorCodes::ERROR_IMPORTING_PYTHON_MODULE);
            return this->storage_ptr->status;
        }
    }
    return this->storage_ptr->status;
}

void CySolverBase::py_diffeq()
{
    // Call the differential equation in python space. Note that the optional arguments are handled by the python 
    // wrapper class. `this->args_ptr` is not used.
    call_diffeq_from_cython(this->cython_extension_class_instance, this->py_diffeq_method);
}
