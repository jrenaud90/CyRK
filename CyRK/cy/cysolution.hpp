#pragma once

#include <cstring>
#include <cstdio>
#include <vector>

#include "common.hpp"

class CySolverResult {

// Attributes
protected:
    // Message storage
    char message[MESSAGE_SIZE] = { };

    // Metadata
    unsigned int num_extra = 0;
    double num_dy_dbl      = 0.0;

    // Current storage information
    size_t original_expected_size = 0;
    size_t storage_capacity       = 0;

    // Storage for arrays
    bool capture_extra = false;

    // Buffer
    unsigned int current_buffer_size = 0;
    double* data_buffer_time_ptr     = &data_buffer_time[0];
    double* data_buffer_y_ptr        = &data_buffer_y[0];

public:
    // Status information
    bool success      = false;
    bool reset_called = false;

    // Error codes
    // 0    : CySolverResult initialized. No error recorded.
    // -999 : Class Error: CySolverResult initialized but correct constructor not called yet.
    // Error codes 1 to 10 and -1 to -10 are defined by CySolver. See the `take_step` method in the base class.
    // -11   : Value Error: Requested new vector size is larger than the limits set by the system (specifically the max of size_t).
    // -12   : Memory Error: Malloc failed when reserving more memory for vectors.
    // -50   : Error calling cython wrapper function from PySolver.
    int error_code = -999;

    // More status information
    char* message_ptr = &message[0];
    size_t size = 0;

    // Metadata
    unsigned int num_y    = 0;
    unsigned int num_dy   = 0;

    // Pointer to storage arrays
    std::vector<double> time_domain = std::vector<double>(0);
    std::vector<double> solution = std::vector<double>(0);

private:
    // Put data buffers at the end of memory stack
    double data_buffer_time[BUFFER_SIZE] = {};
    double data_buffer_y[BUFFER_SIZE * DY_LIMIT] = {};

// Methods
protected:
    void p_expand_storage();
    void p_offload_data();

public:
    CySolverResult();
    CySolverResult(const int num_y, const int num_extra, const size_t expected_size);
    ~CySolverResult();
    void save_data(const double new_t, double* const new_solution_y_ptr, double* const new_solution_dy_ptr);
    void finalize();
    void reset();
    void update_message(const char* const new_message_ptr);

};
