#pragma once

#include <cstring>
#include <vector>
#include <memory>
#include <algorithm>

#include "common.hpp"
#include "dense.hpp"
#include "cy_array.hpp"

class CySolverResult {

// Attributes
protected:
    // Message storage
    char message[MESSAGE_SIZE] = { };

    // Current storage information
    size_t original_expected_size = 0;
    size_t storage_capacity       = 0;
    size_t dense_storage_capacity = 0;

    // Buffer
    unsigned int current_data_buffer_size  = 0;
    unsigned int current_dense_buffer_size = 0;
    double* data_buffer_time_ptr           = &data_buffer_time[0];
    double* data_buffer_y_ptr              = &data_buffer_y[0];
    double* data_buffer_interp_time_ptr    = &data_buffer_interp_time[0];
    CySolverDense** data_buffer_dense_ptr  = &data_buffer_dense_output[0];
    
    // Metadata
    double last_t          = 0;
    double num_dy_dbl      = 0.0;
    unsigned int num_extra = 0;

public:
    // Storage for arrays
    bool capture_extra = false;
    
    // Dense Output
    bool capture_dense_output = false;
    bool t_eval_provided      = false;

    // Status information
    bool success      = false;
    bool reset_called = false;

    // Integration direction
    bool direction_flag = true;

    // Error codes
    // 0    : CySolverResult initialized. No error recorded.
    // -999 : Class Error: CySolverResult initialized but correct constructor not called yet.
    // Error codes 1 to 10 and -1 to -10 are defined by CySolver. See the `take_step` method in the base class.
    // -11   : Value Error: Requested new vector size is larger than the limits set by the system (specifically the max of size_t).
    // -12   : Memory Error: Malloc failed when reserving more memory for vectors.
    // -50   : Error calling cython wrapper function from PySolver.
    int error_code = -999;

    // Metadata
    unsigned int num_y    = 0;
    unsigned int num_dy   = 0;

    // More status information
    char* message_ptr = &message[0];
    size_t size = 0;
    size_t num_interpolates = 0;

    // Pointer to storage arrays
    std::vector<double> time_domain        = std::vector<double>(0);
    std::vector<double> time_domain_sorted = std::vector<double>(0);
    std::vector<double> solution           = std::vector<double>(0);
    double* time_domain_sorted_ptr         = time_domain.data();

    // Dense output array
    std::vector<CySolverDense*> dense_vector = std::vector<CySolverDense*>(0);

    // Interpolant time array (used if t_eval is provided)
    std::vector<double> interp_time = std::vector<double>(0);

private:
    // Put data buffers at the end of memory stack
    double data_buffer_time[BUFFER_SIZE]                 = { };
    double data_buffer_y[BUFFER_SIZE * DY_LIMIT]         = { };
    CySolverDense* data_buffer_dense_output[BUFFER_SIZE] = { };
    double data_buffer_interp_time[BUFFER_SIZE]          = { };


// Methods
protected:
    void p_expand_data_storage();
    void p_expand_dense_storage();
    void p_offload_data();
    void p_offload_dense();

public:
    virtual ~CySolverResult();
    CySolverResult();
    CySolverResult(
        const int num_y,
        const int num_extra,
        const size_t expected_size,
        const double last_t,
        const bool direction_flag,
        const bool capture_dense_output,
        const bool t_eval_provided);
    void save_data(const double new_t, double* const new_solution_y_ptr, double* const new_solution_dy_ptr);
    void save_dense(const double sol_t, CySolverDense* dense_output_ptr);
    void finalize();
    void reset();
    void update_message(const char* const new_message_ptr);
    void call(const double t, double* y_interp_ptr);
    void call_vectorize(const double* t_array_ptr, size_t len_t, double* y_interp_ptr);
};
