#pragma once

#include <memory>

#include "common.hpp"
#include "cysolution.hpp"

void baseline_cysolve_ivp_noreturn(
    std::shared_ptr<CySolverResult> solution_sptr,
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const size_t num_y,
    const int method,
    // General optional arguments
    const size_t expected_size,
    const size_t num_extra,
    const char* args_ptr,
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
);

std::shared_ptr<CySolverResult> baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const size_t num_y,
    const int method,
    // General optional arguments
    const size_t expected_size,
    const size_t num_extra,
    const char* args_ptr,
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
);


/* Pure Python hook solvers and helpers */

class PySolver
{
public:
    // State information
    int status = -999;

    // Integrator information
    int integration_method = -1;

    // Solution information
    std::shared_ptr<CySolverResult> solution_sptr = nullptr;

public:
    PySolver();
    ~PySolver();
    PySolver(
        int integration_method,
        // Cython class instance used for pyhook
        PyObject* cython_extension_class_instance,
        DiffeqMethod cython_extension_class_diffeq_method,
        // Regular integrator inputs
        std::shared_ptr<CySolverResult> solution_sptr,
        const double t_start,
        const double t_end,
        const double* y0_ptr,
        const size_t num_y,
        // General optional arguments
        const size_t expected_size,
        const size_t num_extra,
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
