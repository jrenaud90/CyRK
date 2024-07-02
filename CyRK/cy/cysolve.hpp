#pragma once

#include <memory>

#include "rk.hpp"
#include "cysolver.hpp"


std::shared_ptr<CySolverResult> cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    double* t_span_ptr,
    double* y0_ptr,
    size_t num_y,
    int method,
    // General optional arguments
    size_t expected_size = 0,
    bool capture_extra = false,
    size_t num_extra = 0,
    double* args_ptr = nullptr,
    // rk optional arguments
    size_t max_num_steps = 0,
    size_t max_ram_MB = 2000,
    double rtol = 1.0e-3,
    double atol = 1.0e-6,
    double* rtols_ptr = nullptr,
    double* atols_ptr = nullptr,
    double max_step_size = MAX_STEP,
    double first_step_size = 0.0
);