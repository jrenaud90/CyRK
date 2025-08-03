#pragma once

#include <memory>

#include "common.hpp"
#include "cysolution.hpp"

void baseline_cysolve_ivp_noreturn(
    CySolverResult* solution_ptr,
    DiffeqFuncType diffeq_ptr,
    double t_start,
    double t_end,
    std::vector<double> y0_vec,
    // General optional arguments
    std::optional<size_t> expected_size           = std::nullopt,
    std::optional<size_t> num_extra               = std::nullopt,
    std::optional<std::vector<char>> args_vec     = std::nullopt,
    std::optional<size_t> max_num_steps           = std::nullopt,
    std::optional<size_t> max_ram_MB              = std::nullopt,
    std::optional<bool> capture_dense_output      = std::nullopt,
    std::optional<std::vector<double>> t_eval_vec = std::nullopt,
    std::optional<PreEvalFunc> pre_eval_func      = std::nullopt,
    std::optional<std::vector<double>> rtols      = std::nullopt,
    std::optional<std::vector<double>> atols      = std::nullopt,
    std::optional<double> max_step_size           = std::nullopt,
    std::optional<double> first_step_size         = std::nullopt
);

std::unique_ptr<CySolverResult> baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    double t_start,
    double t_end,
    std::vector<double> y0_vec,
    ODEMethod integration_method,
    // General optional arguments
    std::optional<size_t> expected_size           = std::nullopt,
    std::optional<size_t> num_extra               = std::nullopt,
    std::optional<std::vector<char>> args_vec     = std::nullopt,
    std::optional<size_t> max_num_steps           = std::nullopt,
    std::optional<size_t> max_ram_MB              = std::nullopt,
    std::optional<bool> capture_dense_output      = std::nullopt,
    std::optional<std::vector<double>> t_eval_vec = std::nullopt,
    std::optional<PreEvalFunc> pre_eval_func      = std::nullopt,
    std::optional<std::vector<double>> rtols      = std::nullopt,
    std::optional<std::vector<double>> atols      = std::nullopt,
    std::optional<double> max_step_size           = std::nullopt,
    std::optional<double> first_step_size         = std::nullopt
);
