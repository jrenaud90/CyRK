/* Methods to store and retrieve data saved by CySolver */
#include "cysolution.hpp"

/* ========================================================================= */
/* =========================  Constructors  ================================ */
/* ========================================================================= */
CySolverResult::CySolverResult()
{
    // No cysolver object will be initialized in the base constructor.
    this->solver_uptr = nullptr;
    this->update_status(CyrkErrorCodes::UNINITIALIZED_CLASS);
}

CySolverResult::CySolverResult(ODEMethod integration_method_) :
        integrator_method(integration_method_)
{
    // The result constructor's only job is to build the solver object and allocate its memory.
    CyrkErrorCodes solver_build_status = this->p_build_solver();
    this->update_status(solver_build_status);
}

/* ========================================================================= */
/* =========================  Deconstructors  ============================== */
/* ========================================================================= */
CySolverResult::~CySolverResult()
{
    // Clear all data and reset the size of the vectors. 
    this->dense_vec.clear();
    this->interp_time_vec.clear();
    this->time_domain_vec.clear();
    this->time_domain_vec_sorted.clear();
    this->solution.clear();

    // Deallocate other heap objects.
    // If the deconstructor is called we want to deallocate the solver even if we were asked to retain it before.
    this->solver_uptr.reset();
    this->config_uptr.reset();
}

/* ========================================================================= */
/* =========================  Protected Methods  =========================== */
/* ========================================================================= */
CyrkErrorCodes CySolverResult::p_build_solver()
{
    CyrkErrorCodes build_status = CyrkErrorCodes::NO_ERROR;

    if (this->solver_uptr)
    {
        // Delete old solver.
        this->solver_uptr.reset();
    }

    // The result constructor's only job is to build the solver object and allocate its memory.
    try
    {
        switch (this->integrator_method)
        {
        case ODEMethod::RK23:
            // RK23
            this->solver_uptr = std::make_unique<RK23>(this);
            // this->config_uptr = std::make_unique<RKConfig>(); // We do not currently need to do this since by default we initialize to a RKConfig.
            break;
        case ODEMethod::RK45:
            // RK45
            this->solver_uptr = std::make_unique<RK45>(this);
            // this->config_uptr = std::make_unique<RKConfig>(); // We do not currently need to do this since by default we initialize to a RKConfig.
            break;
        case ODEMethod::DOP853:
            // DOP853
            this->solver_uptr = std::make_unique<DOP853>(this);
            // this->config_uptr = std::make_unique<RKConfig>(); // We do not currently need to do this since by default we initialize to a RKConfig.
            break;
        [[unlikely]] default:
            this->solver_uptr = nullptr;
            this->config_uptr = std::make_unique<ProblemConfig>();
            build_status = CyrkErrorCodes::UNSUPPORTED_UNKNOWN_MODEL;
            break;
        }
    }
    catch (std::bad_alloc const&)
    {
        // Memory allocation failed, return error code
        build_status = CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
    }
    return build_status;
}

void CySolverResult::p_expand_data_storage()
{
    double new_storage_size_dbl = std::floor(DYNAMIC_GROWTH_RATE * (double)this->storage_capacity);

    // Check if this new size is okay.
    if ((new_storage_size_dbl / this->config_uptr->num_dy_dbl) > SIZE_MAX_DBL) [[unlikely]]
    {
        this->update_status(CyrkErrorCodes::VECTOR_SIZE_EXCEEDS_LIMITS);
    }
    else
    {
        this->storage_capacity = (size_t)new_storage_size_dbl;
        // Ensure there is enough new room for the new size.
        this->storage_capacity = std::max<size_t>(this->storage_capacity, std::max<size_t>(this->size, this->num_interpolates) + 1);
        round_to_2(this->storage_capacity);
        try
        {
            this->time_domain_vec.reserve(this->storage_capacity);
            this->solution.reserve(this->storage_capacity * this->num_dy);
            if (this->capture_dense_output)
            {
                this->dense_vec.reserve(this->storage_capacity);
            }
            if (this->t_eval_provided)
            {
                this->interp_time_vec.reserve(this->storage_capacity);
            }
        }
        catch (std::bad_alloc const&)
        {
            this->update_status(CyrkErrorCodes::MEMORY_ALLOCATION_ERROR);
        }
    }
}

void CySolverResult::p_finalize()
{
    // Shrink vectors
    if (this->size > 1000000)
    {
        // There is a lot of data. Let's make sure the vectors are not eating up too much memory if its not needed.
        // If there is not that much data then its not worth the performance hit of shrinking the arrays.
        this->time_domain_vec.shrink_to_fit();
        this->solution.shrink_to_fit();
    }

    if (this->num_interpolates > 1000000)
    {
        if (this->capture_dense_output)
        {
            this->dense_vec.shrink_to_fit();
        }
        if (this->t_eval_provided)
        {
            this->interp_time_vec.shrink_to_fit();
        }
    }

    // Create a sorted time domain
    if (direction_flag)
    {
        // Forward integration. We are already sorted.
        if (this->t_eval_provided)
        {
            this->time_domain_vec_sorted_ptr = &this->interp_time_vec;
        }
        else
        {
            this->time_domain_vec_sorted_ptr = &this->time_domain_vec;
        }
    }
    else
    {
        // Not sorted. Reverse time domain into new storage.
        if (this->t_eval_provided)
        {
            this->time_domain_vec_sorted.resize(this->interp_time_vec.size());
            std::reverse_copy(this->interp_time_vec.begin(), this->interp_time_vec.end(), this->time_domain_vec_sorted.begin());
        }
        else
        {
            this->time_domain_vec_sorted.resize(this->time_domain_vec.size());
            std::reverse_copy(this->time_domain_vec.begin(), this->time_domain_vec.end(), this->time_domain_vec_sorted.begin());
        }
        this->time_domain_vec_sorted_ptr = &this->time_domain_vec_sorted;
    }

    // Check if the integrator finished
    if (this->status == CyrkErrorCodes::SUCCESSFUL_INTEGRATION)
    {
        this->success = true;
    }

    // Delete the solver if we don't need it anymore
    if ((not this->retain_solver) and this->solver_uptr)
    {
        // Make sure that any dense outputs also have their ptr's nulled.
        this->dense_vec.resize(0);

        // Reset the cysolver smart pointer in this class.
        this->solver_uptr.reset();
    }

    // Set the setup variable to false so that subsequent calls will reset the solution.
    this->setup_called = false;
}

/* ========================================================================= */
/* =========================  Public Methods  ============================== */
/* ========================================================================= */
void CySolverResult::update_status(CyrkErrorCodes status_code)
{
    if (status_code != this->status)
    {
        this->status = status_code;
        this->message = CyrkErrorMessages.at(status_code);
    }
}

CyrkErrorCodes CySolverResult::setup()
{
    return this->setup(nullptr);
}

CyrkErrorCodes CySolverResult::setup(ProblemConfig* provided_config_ptr)
{
    this->update_status(CyrkErrorCodes::INITIALIZING);
    CyrkErrorCodes setup_status = CyrkErrorCodes::NO_ERROR;

    while (setup_status == CyrkErrorCodes::NO_ERROR)
    {
        // Reset trackers
        this->size             = 0;
        this->num_interpolates = 0;
        this->steps_taken      = 0;

        // Reset flags
        this->setup_called   = false;
        this->success        = false;
        this->retain_solver  = false;

        // Reset all vectors
        // We resize these to zero so that we can retain any capacity they had from previous runs.
        this->time_domain_vec.resize(0);
        this->time_domain_vec_sorted.resize(0);
        this->solution.resize(0);
        this->interp_time_vec.resize(0);
        this->dense_vec.resize(0);
        this->event_times.resize(0);
        this->event_states.resize(0);

        // Store properties
        if (provided_config_ptr)
        {
            // Check if config is properly setup.
            if (provided_config_ptr->y0_vec.size() == 0)
            {
                setup_status = CyrkErrorCodes::BAD_CONFIG_DATA;
                break;
            }
            this->config_uptr->update_properties_from_config(provided_config_ptr);
        }
        
        // Ensure the configuration file is properly initialized.
        this->config_uptr->initialize();

        // Solver may have been cleared if not forced to retain it.
        if (not this->solver_uptr)
        {
            setup_status = this->p_build_solver();
        }
        // Check if the solver was built successfully.
        if (setup_status != CyrkErrorCodes::NO_ERROR)
        {
            break;
        }
        
        if (not this->solver_uptr)
        {
            setup_status = CyrkErrorCodes::UNINITIALIZED_CLASS;
            break;
        }
        if (not this->config_uptr->initialized)
        {
            // User may have manually updated the current config so the argument may be null
            setup_status = CyrkErrorCodes::UNINITIALIZED_CLASS;
            break;
        }

        // Store some bools in this class for optimization purposes.
        this->capture_dense_output = this->config_uptr->capture_dense_output;
        this->capture_extra        = this->config_uptr->capture_extra;
        this->t_eval_provided      = this->config_uptr->t_eval_provided;
        this->direction_flag       = (this->config_uptr->t_end - this->config_uptr->t_start) >= 0.0;
        this->num_y                = this->config_uptr->num_y;
        this->num_dy               = this->config_uptr->num_dy;
        this->num_events           = this->config_uptr->events_vec.size();
    
        // TODO STORAGE CAPACITY - use previous itereation if set or change to expected size.
        // Storage capacity from previous runs may be larger than expected size. If
        // that is the case then we will just use it instead of reallocating to a smaller size.         
        // Otherwise use expected size.
        if (this->storage_capacity < this->config_uptr->expected_size)
        {
            this->storage_capacity = this->config_uptr->expected_size;
        }
        // Expand vector capacities now to match expected size.
        try
        {
            this->time_domain_vec.reserve(this->storage_capacity);
            this->solution.reserve(this->storage_capacity * this->config_uptr->num_dy);
            if (this->capture_dense_output)
            {
                this->dense_vec.reserve(this->storage_capacity);
            }
            if (this->t_eval_provided)
            {
                this->interp_time_vec.reserve(this->storage_capacity);
            }
            
            // Try to create event storage if needed.
            if (this->num_events > 0)
            {
                this->event_times.resize(this->num_events);
                this->event_states.resize(this->num_events);
                for (size_t event_i = 0; event_i < this->num_events; event_i++)
                {
                    // Assume that we will have the same number of events as storage (this is likely a way overestimate).
                    this->event_times[event_i].reserve(this->storage_capacity);
                    this->event_states[event_i].reserve(this->storage_capacity * this->num_dy);
                }
            }            
        }
        catch (std::bad_alloc const&)
        {
            setup_status = CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
            break;
        }
            
        // If save dense is true and capture extra is true then the dense solutions need to retain the solver to make
        // additional calls to the diffeq. Make sure that the solution class does not delete the solver during finalization.
        // There could be situations where we want to keep the solver in memory even if we are not capturing dense
        // output and not capturing extra.
        if (this->config_uptr->force_retain_solver or 
            (this->capture_dense_output and this->capture_extra))
        {
            this->retain_solver = true;
        }

        // Update the solver with the new configuration.
        setup_status = this->solver_uptr->setup();
        if (setup_status != CyrkErrorCodes::NO_ERROR)
        {
            break;
        }

        // If the user provided a t_eval vector but is not capturing dense output then we need to
        // create a single dense output that will be used to capture the solution at the interpolated time steps.
        if (this->t_eval_provided and (not this->capture_dense_output))
        {
            this->dense_vec.emplace_back(this->solver_uptr.get(), false);
            this->num_interpolates++;
        }

        break;
    }

    if (setup_status == CyrkErrorCodes::NO_ERROR)
    {
        this->setup_called = true;
    }
    this->update_status(setup_status);
    return this->status;
}


void CySolverResult::save_data(
        const double new_t,
        double* const new_solution_y_ptr,
        double* const new_solution_dy_ptr) noexcept
{
    this->size++;
    if (this->size > this->storage_capacity)
    {
        // There is not enough room in the storage vectors. Expand them.
        // The CyRK solution class will use a dynamic growth rate to expand the storage so we do not use the default vector expansion.
        this->p_expand_data_storage();
    }

    // Save time results
    this->time_domain_vec.push_back(new_t);

    // Save y results
    this->solution.insert(this->solution.end(), new_solution_y_ptr, new_solution_y_ptr + this->num_y);

    if (this->config_uptr->capture_extra)
    {
        // Save extra ouput results
        // Start at the end of y values (Dependent dys) and go to the end of the dy array
        this->solution.insert(this->solution.end(), &new_solution_dy_ptr[this->num_y], new_solution_dy_ptr + this->num_dy);
    }
}

void CySolverResult::save_event_data(
        const size_t event_index,
        const double event_t,
        double* const event_y_ptr,
        double* const event_dy_ptr) noexcept
{
    if ((this->num_events > 0) and (event_index < this->num_events))
    {
        // Save time results
        this->event_times[event_index].push_back(event_t);

        // Save y results
        this->event_states[event_index].insert(this->event_states[event_index].end(), event_y_ptr, event_y_ptr + this->num_y);

        if (this->config_uptr->capture_extra)
        {
            // Save extra ouput results
            // Start at the end of y values (Dependent dys) and go to the end of the dy array
            this->event_states[event_index].insert(this->event_states[event_index].end(), &event_dy_ptr[this->num_y], event_dy_ptr + this->num_dy);
        }
    }
}

void CySolverResult::build_dense(bool save_dense) noexcept
{
    if (not this->solver_uptr) [[unlikely]]
    {
        // Solver is required in order to build dense output.
        this->update_status(CyrkErrorCodes::ATTRIBUTE_ERROR);
    }

    if (save_dense)
    {
        this->num_interpolates++;
        if (this->num_interpolates > this->storage_capacity)
        {
            // There is not enough room in the storage vectors. Expand them.
            this->p_expand_data_storage();
        }

        // We need to heap allocate the dense solution
        this->dense_vec.emplace_back(this->solver_uptr.get(), true);

        // Save interpolated time (if t_eval was provided)
        if (this->t_eval_provided)
        {
            this->interp_time_vec.push_back(this->solver_uptr->t_now);
        }
    }
    else
    {
        // Don't need to save. Use the single dense vector that is on the heap. 
        // Need to update its state to match the current solver state.
        this->dense_vec[0].set_state();
    }
}

CyrkErrorCodes CySolverResult::solve()
{
    CyrkErrorCodes solve_status = CyrkErrorCodes::NO_ERROR;
    if (not this->solver_uptr or (this->status != CyrkErrorCodes::NO_ERROR))
    {
        // Solver is not initialized or the status is not NO_ERROR.
        solve_status = CyrkErrorCodes::UNINITIALIZED_CLASS;
    }

    if (not this->setup_called)
    {
        // Setup has not been called; we need to reset the integrator to a base state so try calling setup.
        this->setup(nullptr);
    }

    if (this->solver_uptr and (this->status == CyrkErrorCodes::NO_ERROR))
    {    
        // Tell the solver to starting solving the problem!
        this->solver_uptr->solve();
        
        // Call the finalizer on the storage class instance.
        // This performs some housekeeping so it should be called even if integration failed.
        this->p_finalize();
    }
    return this->status;
}

CyrkErrorCodes CySolverResult::call(const double t, double* y_interp_ptr)
{
    if (not this->success)
    {
        return CyrkErrorCodes::INTEGRATION_NOT_SUCCESSFUL;
    }
    if (not this->capture_dense_output) [[unlikely]]
    {
        return CyrkErrorCodes::DENSE_OUTPUT_NOT_SAVED;
    }
    if ((not y_interp_ptr) or (t < this->config_uptr->t_start) or (t > this->config_uptr->t_end))
    {
        return CyrkErrorCodes::ARGUMENT_ERROR;
    }

    size_t interp_time_vec_len_touse = 0;
    if (this->t_eval_provided)
    {
        interp_time_vec_len_touse = this->num_interpolates;
    }
    else
    {
        interp_time_vec_len_touse = this->size;
    }
    // SciPy uses np.searchedsorted which as far as I can tell works the same as bibnary search with guess
    // Except that it is searchedsorted is 1 more than binary search.
    // This may only hold if the integration is in the forward direction.
    // TODO: See if this holds for backwards integration and update if needed.
    // Get a guess for binary search
    size_t closest_index;

    // Check if there are any t_eval steps between this new index and the last index.
    // Get lowest and highest indices
    double* time_domain_sorted_ptr = this->time_domain_vec_sorted_ptr->data();
    auto lower_i = std::lower_bound(
        time_domain_sorted_ptr,
        time_domain_sorted_ptr + interp_time_vec_len_touse,
        t) - time_domain_sorted_ptr;
    
    auto upper_i = std::upper_bound(
        time_domain_sorted_ptr,
        time_domain_sorted_ptr + interp_time_vec_len_touse,
        t) - time_domain_sorted_ptr;
        
    if (lower_i == upper_i)
    {
        // Only 1 index came back wrapping the value. See if it is different from before.
        closest_index = lower_i;  // Doesn't matter which one we choose
    }
    else if (this->direction_flag)
    {
        // Two different indicies returned. Since we are working our way from low to high values we want the upper one.
        closest_index = upper_i;
    }
    else
    {
        // Two different indicies returned. Since we are working our way from high to low values we want the lower one.
        closest_index = lower_i;
    }

    // Clean up closest index
    if (this->direction_flag)
    {
        closest_index = std::min<size_t>(std::max<size_t>(closest_index, 0), interp_time_vec_len_touse - 1);
    }
    else
    {
        closest_index = interp_time_vec_len_touse - closest_index - 1;
        closest_index = std::min<size_t>(std::max<size_t>(closest_index, 1), interp_time_vec_len_touse - 1);
    }

    if (closest_index > this->dense_vec.size())
    {
        return CyrkErrorCodes::BOUNDS_ERROR;
    }

    // Call interpolant to update y
    this->dense_vec[closest_index].call(t, y_interp_ptr);
    
    return CyrkErrorCodes::NO_ERROR;
}

CyrkErrorCodes CySolverResult::call_vectorize(const double* t_array_ptr, size_t len_t, double* y_interp_ptr)
{
    double* y_sub_ptr;
    CyrkErrorCodes sub_status = CyrkErrorCodes::NO_ERROR;

    for (size_t i = 0; i < len_t; i++)
    {
        // Assume y is passed as a y0_0, y1_0, y2_0, ... y0_1, y1_1, y2_1, ...
        y_sub_ptr = &y_interp_ptr[this->num_dy * i];

        sub_status = this->call(t_array_ptr[i], y_sub_ptr);
        
        if (sub_status != CyrkErrorCodes::NO_ERROR)
        {
            break;
        }
    }
    return sub_status;
}