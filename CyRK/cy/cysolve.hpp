#pragma once

#include <memory>

#include "c_common.hpp"
#include "cysolution.hpp"
#include "c_events.hpp"

void baseline_cysolve_ivp_noreturn(
    CySolverResult* solution_ptr,
    DiffeqFuncType diffeq_ptr,
    double t_start,
    double t_end,
    std::vector<double>& y0_vec,
    // General optional arguments
    size_t expected_size,
    size_t num_extra,
    std::vector<char>& args_vec,
    size_t max_num_steps,
    size_t max_ram_MB,
    bool capture_dense_output,
    std::vector<double>& t_eval_vec,
    PreEvalFunc pre_eval_func,
    std::vector<Event>& events_vec,
    std::vector<double>& rtols,
    std::vector<double>& atols,
    double max_step_size,
    double first_step_size
);

std::unique_ptr<CySolverResult> baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    double t_start,
    double t_end,
    std::vector<double>& y0_vec,
    ODEMethod integration_method,
    // General optional arguments
    size_t expected_size,
    size_t num_extra,
    std::vector<char>& args_vec,
    size_t max_num_steps,
    size_t max_ram_MB,
    bool capture_dense_output,
    std::vector<double>& t_eval_vec,
    PreEvalFunc pre_eval_func,
    std::vector<Event>& events_vec,
    std::vector<double>& rtols,
    std::vector<double>& atols,
    double max_step_size,
    double first_step_size
);
