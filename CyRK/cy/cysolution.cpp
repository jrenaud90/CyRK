/* Methods to store and retrieve data saved by CySolver */

#include "cysolution.hpp"


// Constructors
CySolverResult::CySolverResult()
{

}

CySolverResult::CySolverResult(
        const int num_y,
        const int num_extra,
        const size_t expected_size,
        const double last_t,
        const bool direction_flag,
        const bool capture_dense_output,
        const bool t_eval_provided) :
            last_t(last_t),
            num_extra(num_extra),
            capture_dense_output(capture_dense_output),
            t_eval_provided(t_eval_provided),
            direction_flag(direction_flag),
            error_code(0),
            num_y(num_y)
{
    // Round expected size and store it.
    this->original_expected_size = expected_size;
    round_to_2(this->original_expected_size);

    // num_dy will be larger than num_y if the user wishes to capture extra output during integration.
    this->capture_extra = this->num_extra > 0;
    this->num_dy        = this->num_y + this->num_extra;
    this->num_dy_dbl    = (double)this->num_dy;

    // Initialize other parameters
    this->update_message("CySolverResult Initialized.");
}


// Deconstructors
CySolverResult::~CySolverResult()
{
    // Vector header inforamtion is stack allocated, no need to delete them.
    // The data itself is heap allocated but the vector class will handle that.

    // Need to delete the heap allocated dense solutions
    if (this->capture_dense_output)
    {
        for (size_t i = 0; i < this->num_interpolates; i++)
        {
            if (this->dense_vector[i]) [[likely]]
            {
                delete this->dense_vector[i];
            }
        }
    }
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
            this->time_domain.reserve(this->storage_capacity);
            this->solution.reserve(this->storage_capacity * this->num_dy);
        }
        catch (std::bad_alloc const&)
        {
            this->error_code = -21;
            this->update_message("Memory Error: Malloc failed when reserving additional memory for storage vectors.");
        }
    }
}

void CySolverResult::p_expand_dense_storage()
{
    double new_storage_size_dbl = std::floor(DYNAMIC_GROWTH_RATE * this->dense_storage_capacity);

    // Check if this new size is okay.
    if ((new_storage_size_dbl) > SIZE_MAX_DBL) [[unlikely]]
    {
        this->error_code = -12;
        this->update_message("Value Error: Requested new vector size is larger than the limits set by the system (specifically the max of size_t).");
    }
    else
    {
        this->dense_storage_capacity = (size_t)new_storage_size_dbl;
        // Ensure there is enough new room for the new size.
        this->dense_storage_capacity = std::max<size_t>(this->dense_storage_capacity, this->num_interpolates + 1);

        round_to_2(this->dense_storage_capacity);
        try
        {
            this->dense_vector.reserve(this->dense_storage_capacity);
            if (this->t_eval_provided)
            {
                this->interp_time.reserve(this->dense_storage_capacity);
            }
        }
        catch (std::bad_alloc const&)
        {
            this->error_code = -22;
            this->update_message("Memory Error: Malloc failed when reserving additional memory for dense vectors.");
        }
    }
}

// Public methods
void CySolverResult::reset()
{
    // Inititalize the storage array
    if (this->reset_called) [[unlikely]]
    {
        // The storage array may have already been set. Delete any data and reset it.
        this->time_domain.clear();
        this->solution.clear();
        if (this->capture_dense_output)
        {
            this->dense_vector.clear();
        }
    }

    // Set the storage size to the original expected size.
    this->storage_capacity       = this->original_expected_size;
    this->dense_storage_capacity = this->original_expected_size;

    // Reserve the memory for the vectors
    try
    {
        this->time_domain.reserve(this->storage_capacity);
        this->solution.reserve(this->storage_capacity * this->num_dy);
        if (this->capture_dense_output)
        {
            this->dense_vector.reserve(this->dense_storage_capacity);
        }
        if (this->t_eval_provided)
        {
            this->interp_time.reserve(this->dense_storage_capacity);
        }
    }
    catch (std::bad_alloc const&)
    {
        this->error_code = -12;
        this->update_message("Memory Error: Malloc failed when reserving initial memory for storage vectors.\n");
    }

    this->reset_called = true;
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
    this->time_domain.insert(this->time_domain.end(), this->data_buffer_time_ptr, this->data_buffer_time_ptr + this->current_data_buffer_size);

    // Save y results and any extra output
    this->solution.insert(this->solution.end(), this->data_buffer_y_ptr, this->data_buffer_y_ptr + (this->num_dy * this->current_data_buffer_size));

    // Reset buffers
    this->current_data_buffer_size = 0;
}

void CySolverResult::p_offload_dense()
{
    /*
    * Saves dense solution interpolants that were temporarily held in buffers to heap-allocated vectors.
    */

    this->num_interpolates += this->current_dense_buffer_size;

    if (this->num_interpolates > this->dense_storage_capacity)
    {
        // There is not enough room in the storage vectors. Expand them.
        this->p_expand_dense_storage();
    }

    // Save dense output interpolants
    this->dense_vector.insert(this->dense_vector.end(), this->data_buffer_dense_ptr, this->data_buffer_dense_ptr + this->current_dense_buffer_size);

    // Offload interpolated times
    if (this->t_eval_provided)
    {
        this->interp_time.insert(this->interp_time.end(), this->data_buffer_interp_time_ptr, this->data_buffer_interp_time_ptr + this->current_dense_buffer_size);
    }

    // Reset buffers
    this->current_dense_buffer_size = 0;
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
    unsigned int stride = this->current_data_buffer_size * this->num_dy;  // We have to stride across num_dy even though we are only saving num_y values.
    std::memcpy(&this->data_buffer_y_ptr[stride], new_solution_y_ptr, sizeof(double) * this->num_y);

    // Save extra
    if (this->num_extra > 0)
    {
        std::memcpy(&this->data_buffer_y_ptr[stride + this->num_y], &new_solution_dy_ptr[this->num_y], sizeof(double) * this->num_extra);
    }

    this->current_data_buffer_size++;
}

void CySolverResult::save_dense(const double sol_t, CySolverDense* dense_output_ptr)
{
    // Check if our data buffer is full
    if (this->current_dense_buffer_size >= BUFFER_SIZE)
    {
        this->p_offload_dense();
    }

    // Save data in buffer
    // Save interpolates
    this->data_buffer_dense_ptr[this->current_dense_buffer_size] = dense_output_ptr;

    // Save interpolated time (if t_eval was provided)
    if (this->t_eval_provided)
    {
        this->data_buffer_interp_time_ptr[this->current_dense_buffer_size] = sol_t;
    }

    this->current_dense_buffer_size++;
}

void CySolverResult::finalize()
{
    // Offload anything in the buffer
    if (this->current_data_buffer_size > 0)
    {
        this->p_offload_data();
    }
    if (this->current_dense_buffer_size > 0)
    {
        this->p_offload_dense();
    }
    
    // Shrink vectors
    if (this->size > 100000)
    {
        // There is a lot of data. Let's make sure the vectors are not eating up too much memory if its not needed.
        // If there is not that much data then its not worth the performance hit of shrinking the arrays.
        this->time_domain.shrink_to_fit();
        this->solution.shrink_to_fit();
    }

    if (this->num_interpolates > 10000)
    {
        if (this->capture_dense_output)
        {
            this->dense_vector.shrink_to_fit();
        }
        if (this->t_eval_provided)
        {
            this->interp_time.shrink_to_fit();
        }
    }

    // Create a sorted time domain
    if (direction_flag)
    {
        // Forward integration. We are already sorted.
        if (this->t_eval_provided)
        {
            this->time_domain_sorted_ptr = this->interp_time.data();
        }
        else
        {
            this->time_domain_sorted_ptr = this->time_domain.data();
        }
    }
    else
    {
        // Not sorted. Reverse time domain into new storage.
        if (this->t_eval_provided)
        {
            this->time_domain_sorted.resize(this->interp_time.size());
            std::reverse_copy(this->interp_time.begin(), this->interp_time.end(), this->time_domain_sorted.begin());
        }
        else
        {
            this->time_domain_sorted.resize(this->time_domain.size());
            std::reverse_copy(this->time_domain.begin(), this->time_domain.end(), this->time_domain_sorted.begin());
        }
        this->time_domain_sorted_ptr = this->time_domain_sorted.data();
    }

    // Check if the integrator finished
    if (this->error_code == 1)
    {
        this->success = true;
    }
}

void CySolverResult::update_message(const char* const new_message_ptr)
{
    std::strcpy(this->message_ptr, new_message_ptr);
}

void CySolverResult::call(const double t, double* y_interp)
{
    if (!this->capture_dense_output) [[unlikely]]
    {
        this->error_code = -80;
        this->update_message("Can not call solution when dense output is not saved.");
    }
    else
    {
        size_t interp_time_len_touse = 0;
        if (this->t_eval_provided)
        {
            interp_time_len_touse = this->num_interpolates;
        }
        else
        {
            interp_time_len_touse = this->size;
        }
        // SciPy uses np.searchedsorted which as far as I can tell works the same as bibnary search with guess
        // Except that it is searchedsorted is 1 more than binary search.
        // This may only hold if the integration is in the forward direction.
        // TODO: See if this holds for backwards integration and update if needed.
        // Get a guess for binary search
        size_t closest_index;

        // Check if there are any t_eval steps between this new index and the last index.
        // Get lowest and highest indices
        auto lower_i = std::lower_bound(
            this->time_domain_sorted_ptr,
            this->time_domain_sorted_ptr + interp_time_len_touse,
            t) - this->time_domain_sorted_ptr;
    
        auto upper_i = std::upper_bound(
            this->time_domain_sorted_ptr,
            this->time_domain_sorted_ptr + interp_time_len_touse,
            t) - this->time_domain_sorted_ptr;
        
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
            closest_index = std::min<size_t>(std::max<size_t>(closest_index, 0), interp_time_len_touse - 1);
        }
        else
        {
            closest_index = interp_time_len_touse - closest_index - 1;
            closest_index = std::min<size_t>(std::max<size_t>(closest_index, 1), interp_time_len_touse - 1);
        }

        // Call interpolant to update y
        this->dense_vector[closest_index]->call(t, y_interp);
    }
}

void CySolverResult::call_vectorize(const double* t_array_ptr, size_t len_t, double* y_interp)
{
    double* y_sub_ptr;

    for (size_t i = 0; i < len_t; i++)
    {
        // Assume y is passed as a y0_0, y1_0, y2_0, ... y0_1, y1_1, y2_1, ...
        y_sub_ptr = &y_interp[this->num_y * i];

        this->call(t_array_ptr[i], y_sub_ptr);
    }
}