#pragma once

#include <cstring>
#include <cstdio>
#include <vector>

#include "common.hpp"


class CySolverResult {


// Attributes
protected:
    // Message storage
    char message[256];
    size_t num_extra = 0;
    double num_dy_dbl = 0.0;

    // Current storage information
    size_t original_expected_size = 0;
    size_t storage_capacity = 0;

    // Storage for arrays
    bool capture_extra = false;

public:
    // Metadata
    size_t num_y = 0;
    size_t num_dy = 0;

    // Status information
    char* message_ptr = &this->message[0];
    bool success = false;
    bool reset_called = false;
    size_t size = 0;

    // Pointer to storage arrays
    std::vector<double> time_domain;
    std::vector<double> solution;

    // Error codes
    // 0    : CySolverResult initialized. No error recorded.
    // -999 : Class Error: CySolverResult initialized but correct constructor not called yet.
    // Error codes 1 to 10 and -1 to -10 are defined by CySolver. See the `take_step` method in the base class.
    // -11   : Value Error: Requested new vector size is larger than the limits set by the system (specifically the max of size_t).
    // -12   : Memory Error: Malloc failed when reserving more memory for vectors.
    int error_code = -999;


// Methods
protected:
    void p_expand_storage();

public:
    CySolverResult();
    CySolverResult(size_t num_y, size_t num_extra, size_t expected_size);
    ~CySolverResult();
    void save_data(double new_t, double* new_solution_y, double* new_solution_dy);
    void reset();
    void update_message(const char* new_message);
};
