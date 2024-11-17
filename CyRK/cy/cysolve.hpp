#pragma once

#include <memory>

#include "common.hpp"
#include "cysolution.hpp"


std::shared_ptr<CySolverResult> baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const unsigned int num_y,
    const unsigned int method,
    // General optional arguments
    const size_t expected_size = 0,
    const unsigned int num_extra = 0,
    const void* args_ptr = nullptr,
    const size_t max_num_steps = 0,
    const size_t max_ram_MB = 2000,
    const bool dense_output = false,
    const double* t_eval = nullptr,
    const size_t len_t_eval = 0,
    PreEvalFunc pre_eval_func = nullptr,
    // rk optional arguments
    const double rtol = 1.0e-3,
    const double atol = 1.0e-6,
    const double* rtols_ptr = nullptr,
    const double* atols_ptr = nullptr,
    const double max_step_size = MAX_STEP,
    const double first_step_size = 0.0
);


/* Pure Python hook solvers and helpers */

class PySolver
{
public:
    // State information
    int status = -999;

    // Integrator information
    unsigned int integration_method = 999;

    // Solution information
    std::shared_ptr<CySolverResult> solution_sptr = nullptr;

public:
    PySolver();
    ~PySolver();
    PySolver(
        unsigned int integration_method,
        // Cython class instance used for pyhook
        PyObject* cython_extension_class_instance,
        DiffeqMethod cython_extension_class_diffeq_method,
        // Regular integrator inputs
        std::shared_ptr<CySolverResult> solution_sptr,
        const double t_start,
        const double t_end,
        const double* y0_ptr,
        const unsigned int num_y,
        // General optional arguments
        const size_t expected_size,
        const unsigned int num_extra,
        const void* args_ptr,
        const size_t max_num_steps,
        const size_t max_ram_MB,
        const bool dense_output,
        const double* t_eval,
        const size_t len_t_eval,
        // rk optional arguments
        const double rtol,
        const double atol,
        const double* rtols_ptr,
        const double* atols_ptr,
        const double max_step_size,
        const double first_step_size
    );
    void solve();
};
