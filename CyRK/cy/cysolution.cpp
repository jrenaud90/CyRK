/* Methods to store and retrieve data saved by CySolver */

#include "cysolution.hpp"


// Constructors
CySolverResult::CySolverResult()
{

}

CySolverResult::CySolverResult(const int num_y, const int num_extra, const size_t expected_size) :
        num_y(num_y),
        num_extra(num_extra),
        error_code(0)        
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
}


// Protected methods
void CySolverResult::p_expand_storage()
{
    double new_storage_size_dbl = std::floor(DYNAMIC_GROWTH_RATE * this->storage_capacity);

    // Check if this new size is okay.
    if ((new_storage_size_dbl / this->num_dy_dbl) > SIZE_MAX_DBL)
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
            this->error_code = -12;
            this->update_message("Memory Error: Malloc failed when reserving additional memory for storage vectors.");
        }
    }
}

// Public methods
void CySolverResult::reset()
{
    // Inititalize the storage array
    if (this->reset_called)
    {
        // The storage array may have already been set. Delete any data and reset it.
        this->time_domain.clear();
        this->solution.clear();
    }

    // Set the storage size to the original expected size.
    this->storage_capacity = this->original_expected_size;

    // Reserve the memory for the vectors
    try
    {
        this->time_domain.reserve(this->storage_capacity);
        this->solution.reserve(this->storage_capacity * this->num_dy);
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
    this->size += this->current_buffer_size;

    if (this->size > this->storage_capacity)
    {
        // There is not enough room in the storage vectors. Expand them.
        this->p_expand_storage();
    }

    // Save time results
    this->time_domain.insert(this->time_domain.end(), this->data_buffer_time_ptr, this->data_buffer_time_ptr + this->current_buffer_size);

    // Save y results and any extra output
    this->solution.insert(this->solution.end(), this->data_buffer_y_ptr, this->data_buffer_y_ptr + (this->num_dy * this->current_buffer_size));

    // Reset buffers
    this->current_buffer_size = 0;
}

void CySolverResult::save_data(const double new_t, double* const new_solution_y_ptr, double* const new_solution_dy_ptr)
{
    // Check if our data buffer is full
    if (this->current_buffer_size >= BUFFER_SIZE)
    {
        this->p_offload_data();
    }

    // Save data in buffer
    // Save time
    this->data_buffer_time_ptr[this->current_buffer_size] = new_t;

    // Save y
    unsigned int stride = this->current_buffer_size * this->num_dy;
    std::memcpy(&this->data_buffer_y_ptr[stride], new_solution_y_ptr, sizeof(double) * this->num_y);

    // Save extra
    if (this->num_extra > 0)
    {
        std::memcpy(&this->data_buffer_y_ptr[stride + this->num_y], &new_solution_dy_ptr[this->num_y], sizeof(double) * this->num_extra);
    }

    this->current_buffer_size++;
}

void CySolverResult::finalize()
{
    // Offload anything in the buffer
    if (this->current_buffer_size > 0)
    {
        this->p_offload_data();
    }
    
    // Shrink vectors
    this->time_domain.shrink_to_fit();
    this->solution.shrink_to_fit();

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

// Getters
std::vector<double> CySolverResult::get_time_domain()
{
    return this->time_domain;
}

std::vector<double> CySolverResult::get_solution()
{
    return this->solution;
}