/* Methods to store and retrieve data saved by CySolver */

#include "cysolution.hpp"


// Constructors
CySolverResult::CySolverResult()
{

}

CySolverResult::CySolverResult(
        const size_t num_y,
        const size_t num_extra,
        const size_t expected_size,
        const double last_t,
        const bool direction_flag,
        const bool capture_dense_output,
        const bool t_eval_provided) :
            last_t(last_t),
            capture_dense_output(capture_dense_output),
            t_eval_provided(t_eval_provided),
            direction_flag(direction_flag),
            error_code(0),
            num_y(num_y),
            num_extra(num_extra)
{
    // num_dy will be larger than num_y if the user wishes to capture extra output during integration.
    this->capture_extra = this->num_extra > 0;
    this->num_dy        = this->num_y + this->num_extra;
    this->num_dy_dbl    = (double)this->num_dy;
    this->set_expected_size(expected_size);

    // If save dense is true and capture extra is true then the dense solutions need to retain the solver to make
    // additional calls to the diffeq. Make sure that the solution class does not delete the solver during finalization.
    if (this->capture_dense_output && this->capture_extra)
    {
        this->retain_solver = true;
    }

    // Initialize other parameters
    this->update_message("CySolverResult Initialized.");
}


// Deconstructors
CySolverResult::~CySolverResult()
{
    this->p_delete_heap();
}

void CySolverResult::p_delete_heap()
{
    // Instruct the heap allocated vectors to start deconstructing.
    this->dense_vec.clear();
    this->interp_time_vec.clear();
    this->time_domain_vec.clear();
    this->time_domain_vec_sorted.clear();
    this->solution.clear();

    // Need to delete the solver
    this->solver_uptr.reset();
}


// Protected methods
void CySolverResult::p_expand_data_storage()
{
    double new_storage_size_dbl = std::floor(DYNAMIC_GROWTH_RATE * this->storage_capacity);

    // Check if this new size is okay.
    if ((new_storage_size_dbl / this->num_dy_dbl) > SIZE_MAX_DBL) [[unlikely]]
    {
        this->error_code = -11;
        this->update_message("Value Error: Requested new vector size is larger than the limits set by the system (specifically the max of size_t).");
    }
    else
    {
        this->storage_capacity = (size_t)new_storage_size_dbl;
        // Ensure there is enough new room for the new size.
        this->storage_capacity = std::max<size_t>(this->storage_capacity, this->size + 1);

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
            this->error_code = -21;
            this->update_message("Memory Error: Malloc failed when reserving additional memory for storage vectors.");
        }
    }
}

void CySolverResult::p_offload_data()
{
    /*
    * Saves integration data that were temporarily held in buffers to heap-allocated vectors.
    */

    this->size += this->current_data_buffer_size;

    if (this->size > this->storage_capacity)
    {
        // There is not enough room in the storage vectors. Expand them.
        this->p_expand_data_storage();
    }

    // Save time results
    this->time_domain_vec.insert(this->time_domain_vec.end(), this->data_buffer_time_ptr, this->data_buffer_time_ptr + this->current_data_buffer_size);

    // Save y results and any extra output
    this->solution.insert(this->solution.end(), this->data_buffer_y_ptr, this->data_buffer_y_ptr + (this->num_dy * this->current_data_buffer_size));

    // Reset buffers
    this->current_data_buffer_size = 0;
}


// Public methods
void CySolverResult::set_expected_size(size_t expected_size)
{
    // Round expected size and store it.
    this->original_expected_size = expected_size;
    round_to_2(this->original_expected_size);
}

void CySolverResult::reset()
{
    // Initialize the storage array
    if (this->reset_called) [[unlikely]]
    {
        // The storage array may have already been set. Delete any data and reset it.
        this->time_domain_vec.clear();
        this->solution.clear();

        // Ensure any previous heap allocated data is deleted
        this->p_delete_heap();
    }

    // Set the storage size to the original expected size.
    this->storage_capacity = this->original_expected_size;

    // Reset buffer trackers
    this->current_data_buffer_size  = 0;

    // Reserve the memory for the vectors
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
        this->error_code = -12;
        this->update_message("Memory Error: Malloc failed when reserving initial memory for storage vectors.\n");
    }

    this->reset_called = true;
}

void CySolverResult::build_solver(
        DiffeqFuncType diffeq_ptr,
        const double t_start,
        const double t_end,
        const double* y0_ptr,
        const int method,
        // General optional arguments
        const size_t expected_size,
        const void* args_ptr,
        const size_t max_num_steps,
        const size_t max_ram_MB,
        const bool dense_output,
        const double* t_eval,
        const size_t len_t_eval,
        PreEvalFunc pre_eval_func,
        // rk optional arguments
        const double rtol,
        const double atol,
        const double* rtols_ptr,
        const double* atols_ptr,
        const double max_step_size,
        const double first_step_size
    )
{
    // Make sure the solver pointer is empty.
    if (this->solver_uptr)
    {
        this->solver_uptr.reset();
    }
    
    this->integrator_method = method;

    switch (this->integrator_method)
    {
    case 0:
        // RK23
        this->solver_uptr = std::make_unique<RK23>(
            // Common Inputs
            diffeq_ptr, this->shared_from_this(), t_start, t_end, y0_ptr,
            this->num_y, this->num_extra, args_ptr, max_num_steps, max_ram_MB,
            this->capture_dense_output, t_eval, len_t_eval, pre_eval_func,
            // RK Inputs
            rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
        );
        break;
    case 1:
        // RK45
        this->solver_uptr = std::make_unique<RK45>(
            // Common Inputs
            diffeq_ptr, this->shared_from_this(), t_start, t_end, y0_ptr, num_y,
            this->num_extra, args_ptr, max_num_steps, max_ram_MB,
            this->capture_dense_output, t_eval, len_t_eval, pre_eval_func,
            // RK Inputs
            rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
        );
        break;
    case 2:
        // DOP853
        this->solver_uptr = std::make_unique<DOP853>(
            // Common Inputs
            diffeq_ptr, this->shared_from_this(), t_start, t_end, y0_ptr, num_y,
            this->num_extra, args_ptr, max_num_steps, max_ram_MB,
            this->capture_dense_output, t_eval, len_t_eval, pre_eval_func,
            // RK Inputs
            rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
        );
        break;
    [[unlikely]] default:
        this->solver_uptr.reset();
        this->success     = false;
        this->error_code = -3;
        this->update_message("Model Error: Not implemented or unknown CySolver model requested.\n");
        break;
    }

    // Setup a single heap allocated dense solver if it is needed.
    if (t_eval_provided && !this->capture_dense_output)
    {
        this->dense_vec.emplace_back(this->integrator_method, this->solver_uptr.get(), false);
        this->num_interpolates++;
    }
}

void CySolverResult::save_data(const double new_t, double* const new_solution_y_ptr, double* const new_solution_dy_ptr)
{
    // Check if our data buffer is full
    if (this->current_data_buffer_size >= BUFFER_SIZE)
    {
        this->p_offload_data();
    }

    // Save data in buffer
    // Save time
    this->data_buffer_time_ptr[this->current_data_buffer_size] = new_t;

    // Save y
    size_t stride = this->current_data_buffer_size * this->num_dy;  // We have to stride across num_dy even though we are only saving num_y values.
    std::memcpy(&this->data_buffer_y_ptr[stride], new_solution_y_ptr, sizeof(double) * this->num_y);

    // Save extra
    if (this->num_extra > 0)
    {
        std::memcpy(&this->data_buffer_y_ptr[stride + this->num_y], &new_solution_dy_ptr[this->num_y], sizeof(double) * this->num_extra);
    }

    this->current_data_buffer_size++;
}

CySolverDense* CySolverResult::build_dense(bool save)
{
    if (!this->solver_uptr) [[unlikely]]
    {
        return nullptr;
    }

    if (save)
    {
        this->num_interpolates++;
        if (this->num_interpolates > this->storage_capacity)
        {
            // There is not enough room in the storage vectors. Expand them.
            this->p_expand_data_storage();
        }

        // We need to heap allocate the dense solution
        this->dense_vec.emplace_back(this->integrator_method, this->solver_uptr.get(), true);

        // Save interpolated time (if t_eval was provided)
        if (this->t_eval_provided)
        {
            this->interp_time_vec.push_back(this->solver_uptr->t_now);
        }

        return &this->dense_vec.back();
    }
    else
    {
        // Don't need to save. Use the single dense vector that is on the heap. 
        // Need to update its state to match the current solver state.
        this->dense_vec[0].set_state();

        return &this->dense_vec[0];
    }
}

void CySolverResult::solve()
{
    if (this->solver_uptr)
    {
        // Reset the solver back to t=0
        this->solver_uptr->reset();

        // Tell the solver to starting solving the problem
        this->solver_uptr->solve();
        
        // Call the finalizer on the storage class instance
        this->finalize();
    }
}

void CySolverResult::finalize()
{
    // Offload anything in the buffer
    if (this->current_data_buffer_size > 0)
    {
        this->p_offload_data();
    }

    // Shrink vectors
    if (this->size > 100000)
    {
        // There is a lot of data. Let's make sure the vectors are not eating up too much memory if its not needed.
        // If there is not that much data then its not worth the performance hit of shrinking the arrays.
        this->time_domain_vec.shrink_to_fit();
        this->solution.shrink_to_fit();
    }

    if (this->num_interpolates > 10000)
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
    if (this->error_code == 1)
    {
        this->success = true;
    }

    // Delete the solver if we don't need it anymore
    if (!this->retain_solver && this->solver_uptr)
    {
        // Reset the cysolver smart pointer in this class.
        this->solver_uptr.reset();
    }
}

void CySolverResult::update_message(const char* const new_message_ptr)
{
    std::strcpy(this->message_ptr, new_message_ptr);
}

void CySolverResult::call(const double t, double* y_interp_ptr)
{
    if (!this->capture_dense_output) [[unlikely]]
    {
        this->error_code = -80;
        this->update_message("Can not call solution when dense output is not saved.");
    }
    else
    {
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

        // Call interpolant to update y
        this->dense_vec[closest_index].call(t, y_interp_ptr);
    }
}

void CySolverResult::call_vectorize(const double* t_array_ptr, size_t len_t, double* y_interp_ptr)
{
    double* y_sub_ptr;

    for (size_t i = 0; i < len_t; i++)
    {
        // Assume y is passed as a y0_0, y1_0, y2_0, ... y0_1, y1_1, y2_1, ...
        y_sub_ptr = &y_interp_ptr[this->num_dy * i];

        this->call(t_array_ptr[i], y_sub_ptr);
    }
}