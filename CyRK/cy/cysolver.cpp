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
        bool force_retain_solver_,
        std::vector<Event>& events_vec_): 
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
            force_retain_solver(force_retain_solver_),
            events_vec(events_vec_)
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
        bool force_retain_solver_,
        std::vector<Event>& events_vec_)
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
    this->events_vec = events_vec_;

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
    this->check_events    = this->events_vec.size() > 0;
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
        new_config_ptr->force_retain_solver,
        new_config_ptr->events_vec
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
        this->dy_holder_vec.resize(num_dy_ * 4); // 4 is the number of subarrays held in this vector.
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
    this->dy_tmp2_ptr     = &dy_holder_ptr[num_dy_ * 3];

    return CyrkErrorCodes::NO_ERROR;
}

CyrkErrorCodes CySolverBase::setup()
{
    CyrkErrorCodes setup_status = CyrkErrorCodes::NO_ERROR;

    // Reset flags
    this->t_eval_finished   = false;
    this->setup_called      = false;
    this->error_flag        = false;
    this->check_events_flag = false;
    this->num_events        = 0;
    this->user_provided_max_num_steps = false;
    this->clear_python_refs();
    this->event_data_vec.resize(0);
    this->active_event_indices_vec.resize(0);

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
        this->num_y             = this->storage_ptr->config_uptr->num_y;
        this->num_extra         = this->storage_ptr->config_uptr->num_extra;
        this->num_dy            = this->storage_ptr->config_uptr->num_dy;
        this->sizeof_dbl_Ny     = sizeof(double) * this->num_y;
        this->sizeof_dbl_Ndy    = sizeof(double) * this->num_dy;
        this->num_y_dbl         = this->storage_ptr->config_uptr->num_y_dbl;
        this->num_y_sqrt        = this->storage_ptr->config_uptr->num_y_sqrt;
        this->capture_extra     = this->num_extra > 0;
        this->use_dense_output  = this->storage_ptr->config_uptr->capture_dense_output;
        this->check_events_flag = this->storage_ptr->config_uptr->check_events;
        this->num_events        = this->storage_ptr->config_uptr->events_vec.size();

        // Setup time information
        this->len_t            = 0;
        this->t_start          = this->storage_ptr->config_uptr->t_start;
        this->t_end            = this->storage_ptr->config_uptr->t_end;
        this->t_delta          = this->t_end - this->t_start;
        this->t_delta_abs      = std::fabs(this->t_delta);
        this->direction_flag   = this->t_delta >= 0.0;
        this->direction_inf    = (this->direction_flag) ? INF : -INF;
        this->termination_root = this->direction_inf;

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

        // Setup event storage
        if (this->check_events_flag)
        {
            try
            {
                this->root_finder_data.y_vec.resize(2 * this->num_dy);
                this->event_data_vec.resize(this->num_events);
                this->active_event_indices_vec.reserve(this->num_events);
            }
            catch (const std::bad_alloc&)
            {
                setup_status = CyrkErrorCodes::MEMORY_ALLOCATION_ERROR;
                break;
            }
            this->event_checks_old_ptr = this->event_data_vec.data();

            // Reset the event counters on each event.
            for (size_t event_i = 0; event_i < this->num_events; event_i++)
            {
                Event& current_event = this->storage_ptr->config_uptr->events_vec[event_i];
                current_event.current_count = 0;
                current_event.last_root     = NAN;
                current_event.is_active     = false;
                // Setup event storages
                current_event.y_at_root_vec.resize(this->num_dy);
            }
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

        // Initialize event check data
        if (this->check_events_flag)
        {
            double* event_y_now_use_ptr = this->y_now_ptr;
            if (this->capture_extra)
            {
                // If we are capturing extra variables then we want to pass those
                // to the event functions as well.

                // We will use dy_tmp2_ptr to hold the combined y and extra values.
                // dy_tmp_ptr is used during dense calls so do not want to rely on that memory not being overwritten
                // during the event loop. 
                // dy_tmp2_ptr is the correct size. But the first num_y values need to be copied over from y_now_ptr.
                std::memcpy(this->dy_tmp2_ptr, this->y_now_ptr, this->sizeof_dbl_Ny);
                // The rest come from dy_now_ptr.
                std::memcpy(&this->dy_tmp2_ptr[this->num_y], &this->dy_now_ptr[this->num_y], sizeof(double) * this->num_extra);
                event_y_now_use_ptr = this->dy_tmp2_ptr;
            }

            for (size_t event_i = 0; event_i < this->num_events; event_i++)
            {
                Event& current_event = this->storage_ptr->config_uptr.get()->events_vec[event_i];
    
                // Find new event state array
                this->event_checks_old_ptr[event_i] = current_event.check(
                    this->t_now,
                    event_y_now_use_ptr,
                    this->args_ptr);
            }
        }        

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
        return 
                (this->storage_ptr->status == CyrkErrorCodes::NO_ERROR) // This will be false if there is an error or if integration is complete.
            and (not this->error_flag)
            and this->setup_called;
    }
    return false;
}

CyrkErrorCodes CySolverBase::p_check_events() noexcept
{
    // Pull out event vector and dense reference
    Event* event_ptr = this->storage_ptr->config_uptr.get()->events_vec.data();
    CySolverDense* const dense_func_ptr = &this->storage_ptr->dense_vec.back();

    // Root finding parameters
    const double BRENTQ_ATOL     = 4.0 * EPS;
    const double BRENTQ_RTOL     = 4.0 * EPS;
    const size_t MAX_BRENTQ_ITER = 100;

    double* event_y_now_use_ptr = this->y_now_ptr;
    if (this->capture_extra)
    {
        // If we are capturing extra variables then we want to pass those
        // to the event functions as well.

        // We will use dy_tmp2_ptr to hold the combined y and extra values.
        // dy_tmp_ptr is used during dense calls so do not want to rely on that memory not being overwritten
        // during the event loop. 
        // dy_tmp2_ptr is the correct size. But the first num_y values need to be copied over from y_now_ptr.
        std::memcpy(this->dy_tmp2_ptr, this->y_now_ptr, this->sizeof_dbl_Ny);
        // The rest come from dy_now_ptr.
        std::memcpy(&this->dy_tmp2_ptr[this->num_y], &this->dy_now_ptr[this->num_y], sizeof(double) * this->num_extra);
        event_y_now_use_ptr = this->dy_tmp2_ptr;
    }
    
    // Reset any previously active events and find if any are active now.
    this->active_event_indices_vec.resize(0);

    // We want to stop storing data at the root of a triggered termination event (if there are any).
    // If we are doing forward integration we want the smallest root. 
    // If we are doing backward integration we want the largest root.
    this->termination_root = this->direction_inf;

    for (size_t event_i = 0; event_i < this->num_events; event_i++)
    {
        Event& current_event = event_ptr[event_i];
    
        // Find new event state array
        double g_now = current_event.check(
            this->t_now,
            event_y_now_use_ptr,
            this->args_ptr);
        
        // Check if event was triggered
        // This section mimics scipy's `find_active_events` function.
        double g_old = this->event_checks_old_ptr[event_i];
        bool event_triggered = false;
        
        // Check if event was triggered by looking at the past and current event results.
        // User can specify if they only want an event to trigger when approaching zero from
        // above (direction > 0) or below (direction < 0) or either (direction == 0).
        bool up     = (g_old <= 0.0) and (g_now >= 0.0);
        bool down   = (g_old >= 0.0) and (g_now <= 0.0);
        bool either = up or down;
        if ((current_event.direction == 0) and either)
        {
            event_triggered = true;
        }
        else if ((current_event.direction > 0) and up)
        {
            event_triggered = true;
        }
        else if ((current_event.direction < 0) and down)
        {
            event_triggered = true;
        }

        // update old event check value
        this->event_checks_old_ptr[event_i] = g_now;

        if (not event_triggered)
        {
            // If event was not triggered then continue to next event.
            current_event.is_active = false;
            continue;
        }

        // Event was triggered
        current_event.is_active = true;
        current_event.current_count++;
        this->active_event_indices_vec.push_back(event_i);
        // Reset root data from any previous calls.
        this->root_finder_data.funcalls   = 0;
        this->root_finder_data.iterations = 0;
        this->root_finder_data.error_num  = CyrkErrorCodes::NO_ERROR;
        
        // Below mimics scipy's event `handle_events` function.
        // Find the root of the event function using the BrentQ method.
        current_event.last_root = c_brentq(
            current_event.check,
            this->t_old,
            this->t_now,
            BRENTQ_ATOL,
            BRENTQ_RTOL,
            MAX_BRENTQ_ITER,
            this->storage_ptr->config_uptr->args_vec,
            &this->root_finder_data,
            dense_func_ptr);
        
        if (root_finder_data.error_num != CyrkErrorCodes::CONVERGED)
        {
            // Root finding failed.
            current_event.status = root_finder_data.error_num;
            this->error_flag = true;
            return root_finder_data.error_num;
        }

        // The root finder also finds the y values (both dependent and extra if applicable) at the root.
        // Store these values in the event structure.
        if (root_finder_data.y_at_root_ptr == nullptr) [[unlikely]]
        {
            this->error_flag = true;
            return CyrkErrorCodes::ATTRIBUTE_ERROR;
        }
        std::memcpy(
            current_event.y_at_root_vec.data(),
            root_finder_data.y_at_root_ptr,
            this->sizeof_dbl_Ndy);
        
        if (current_event.current_count >= current_event.max_allowed)
        {
            // Termination condition met.
            this->storage_ptr->event_terminated = true;
            current_event.status = CyrkErrorCodes::EVENT_TERMINATED;
            // Find the smallest/largest root depending on integration direction.
            if (this->direction_flag)
            {
                // Forward integration
                if (current_event.last_root < this->termination_root)
                {
                    this->termination_root = current_event.last_root;
                }
            }
            else
            {
                // Backward integration
                if (current_event.last_root > this->termination_root)
                {
                    this->termination_root = current_event.last_root;
                }
            }
        }
    }

    // Loop through active events and disable any that are beyond the termination root.
    if (this->storage_ptr->event_terminated)
    {
        // We are using this convoluted looping so we can erase active events as we go.
        for (auto iter = this->active_event_indices_vec.begin(); iter != this->active_event_indices_vec.end(); )
        {
            size_t event_i = *iter;
            Event& current_event = event_ptr[event_i];
            
            if (current_event.last_root == this->termination_root)
            {
                // This was the event (or one of the events) that caused the termination. Record its index.
                this->storage_ptr->event_terminate_index = event_i;
            }

            if (this->direction_flag)
            {
                // Forward integration
                if (current_event.last_root > this->termination_root)
                {
                    current_event.is_active = false;
                    iter = this->active_event_indices_vec.erase(iter);
                }
                else
                {
                    ++iter;
                }
            }
            else
            {
                // Backward integration
                if (current_event.last_root < this->termination_root)
                {
                    current_event.is_active = false;
                    iter = this->active_event_indices_vec.erase(iter);
                }
                else
                {
                    ++iter;
                }
            }
        }
    }
    return CyrkErrorCodes::NO_ERROR;
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

        if (this->check_events_flag)
        {
            if (not dense_built)
            {
                // We need an interpolator to check events. If it is not built already then build it now.
                // The `false` flag tells the storage to not to append the interpolator. 
                // One interpolated will be overwritten at each step.
                this->storage_ptr->build_dense(false);
                dense_built = true;
            }

            // Check events
            this->storage_ptr->update_status(this->p_check_events());

            // Save event data
            this->storage_ptr->record_event_data();

            // Check for termination condition
            if (this->storage_ptr->event_terminated and (not this->error_flag))
            {
                // Update integration status since we are now done.
                this->storage_ptr->update_status(CyrkErrorCodes::EVENT_TERMINATED);

                // We want to set t_now to the termination root so all data is saved up until termination.
                if (this->t_now != this->termination_root)
                {
                    this->t_now = this->termination_root;

                    // Also need to update y_now and dy_now to reflect this new time.
                    // Dense output is guaranteed to be built since its needed by event checker.
                    this->storage_ptr->dense_vec.back().call(this->t_now, this->y_now_ptr);

                    if (this->capture_extra)
                    {
                        // Call diffeq to get new dy_now values where extra output is stored.
                        this->diffeq(this);
                    }
                }
            }
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
    if (
           (this->storage_ptr->status == CyrkErrorCodes::SUCCESSFUL_INTEGRATION)
        or (this->storage_ptr->status == CyrkErrorCodes::EVENT_TERMINATED)
       )
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
