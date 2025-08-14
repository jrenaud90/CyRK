#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <algorithm>

#include "common.hpp"
#include "cy_array.hpp"
#include "dense.hpp"
#include "cysolver.hpp"
#include "rk.hpp"


class CySolverResult {

// Attributes
protected:
    // Current storage information
    size_t storage_capacity = 0;

public:
    // Message storage
    std::string message;

    // Configurations - Currently we will always be making a RK config so let's initialize to that for now.
    std::unique_ptr<ProblemConfig> config_uptr = std::make_unique<RKConfig>();

    // Current storage information
    ODEMethod integrator_method = ODEMethod::NO_METHOD_SET;
    CyrkErrorCodes status = CyrkErrorCodes::UNSET_ERROR_CODE;
    size_t size             = 0;
    size_t num_interpolates = 0;
    size_t steps_taken      = 0; // Steps taken will be different than size if the user provided a t_eval vector.

    // Status information
    bool setup_called = false;
    bool success      = false;

    // Problem-specific flags
    bool retain_solver        = false;
    bool capture_dense_output = false;
    bool capture_extra        = false;
    bool t_eval_provided      = false;
    bool direction_flag       = true;

    // Dependent variable
    size_t num_y  = 0;
    size_t num_dy = 0;

    // Pointer to storage arrays
    // Default construct them with a size of 512.
    std::vector<double> time_domain_vec        = std::vector<double>(PRE_ALLOC_STEPS);
    std::vector<double> solution               = std::vector<double>(PRE_ALLOC_STEPS * PRE_ALLOC_NUMY);
    std::vector<double> time_domain_vec_sorted = std::vector<double>(0);
    std::vector<double>* time_domain_vec_sorted_ptr = nullptr;
    
    // Dense output array
    std::vector<CySolverDense> dense_vec = std::vector<CySolverDense>(0);  // Heap allocated dense solutions for when the user needs these saved.

    // Solver storage
    std::unique_ptr<CySolverBase> solver_uptr = nullptr;

    // Interpolant time array (used if t_eval is provided)
    std::vector<double> interp_time_vec = std::vector<double>(0);

// Methods
protected:
    void p_expand_data_storage();
    void p_finalize();
    CyrkErrorCodes p_build_solver();

public:
    virtual ~CySolverResult();
    CySolverResult();
    CySolverResult(ODEMethod integration_method_);
    
    void update_status(CyrkErrorCodes status_code);
    CyrkErrorCodes setup();
    CyrkErrorCodes setup(ProblemConfig* config_ptr);
    void save_data(
        const double new_t,
        double* const new_solution_y_ptr,
        double* const new_solution_dy_ptr) noexcept;
    void build_dense(bool save_dense) noexcept;
    CyrkErrorCodes solve();
    CyrkErrorCodes call(const double t, double* y_interp_ptr);
    CyrkErrorCodes call_vectorize(const double* t_array_ptr, size_t len_t, double* y_interp_ptr);
};
