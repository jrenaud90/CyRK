#include <stdexcept>

#include "cysolve.hpp"
#include "cysolver.hpp"

void baseline_cysolve_ivp_noreturn(
        CySolverResult* solution_ptr,
        DiffeqFuncType diffeq_ptr,
        double t_start,
        double t_end,
        std::vector<double> y0_vec,
        // General optional arguments
        std::optional<size_t> expected_size,
        std::optional<size_t> num_extra,
        std::optional<std::vector<char>> args_vec,
        std::optional<size_t> max_num_steps,
        std::optional<size_t> max_ram_MB,
        std::optional<bool> capture_dense_output,
        std::optional<std::vector<double>> t_eval_vec,
        std::optional<PreEvalFunc> pre_eval_func,
        // rk optional arguments
        std::optional<std::vector<double>> rtols,
        std::optional<std::vector<double>> atols,
        std::optional<double> max_step_size,
        std::optional<double> first_step_size
    )
{
    // Make sure there is a configuration in the solution.
    RKConfig* rk_config_ptr = nullptr;
    if (not solution_ptr->config_uptr)
    {
        // For now we are only ever using RK methods so it is safe to assume RK Config.
        solution_ptr->config_uptr = std::make_unique<RKConfig>(
            diffeq_ptr,
            t_start,
            t_end,
            y0_vec
        );
        rk_config_ptr = static_cast<RKConfig*>(solution_ptr->config_uptr.get());
        rk_config_ptr->update_properties(
            std::nullopt, // diffeq_ptr Already set during construction.
            num_extra,
            std::nullopt, // t_start Already set during construction.
            std::nullopt, // t_end Already set during construction.
            std::nullopt, // y0_vec Already set during construction.
            args_vec,
            t_eval_vec,
            expected_size,
            max_num_steps,
            max_ram_MB,
            pre_eval_func,
            capture_dense_output,
            std::nullopt, // force_retain_solver; not currently in use.
            rtols,
            atols,
            max_step_size,
            first_step_size
        );
    }
    else
    {
        // For now we are only ever using RK methods so it is safe to assume RK Config.
        rk_config_ptr = static_cast<RKConfig*>(solution_ptr->config_uptr.get());
        // Change the configurations of the solver.
        rk_config_ptr->update_properties(
            diffeq_ptr,
            num_extra,
            t_start,
            t_end,
            y0_vec,
            args_vec,
            t_eval_vec,
            expected_size,
            max_num_steps,
            max_ram_MB,
            pre_eval_func,
            capture_dense_output,
            std::nullopt, // force_retain_solver; not currently in use.
            rtols,
            atols,
            max_step_size,
            first_step_size
        );
    }
    // Initialize the solution and solver given the new configurations
    solution_ptr->setup(nullptr);

    // Run integrator
    solution_ptr->solve();
}

std::unique_ptr<CySolverResult> baseline_cysolve_ivp(
        DiffeqFuncType diffeq_ptr,
        double t_start,
        double t_end,
        std::vector<double> y0_vec,
        ODEMethod integration_method,
        // General optional arguments
        std::optional<size_t> expected_size,
        std::optional<size_t> num_extra,
        std::optional<std::vector<char>> args_vec,
        std::optional<size_t> max_num_steps,
        std::optional<size_t> max_ram_MB,
        std::optional<bool> capture_dense_output,
        std::optional<std::vector<double>> t_eval_vec,
        std::optional<PreEvalFunc> pre_eval_func,
        // rk optional arguments
        std::optional<std::vector<double>> rtols,
        std::optional<std::vector<double>> atols,
        std::optional<double> max_step_size,
        std::optional<double> first_step_size
    )
{
    // Build storage class
    std::unique_ptr<CySolverResult> solution_uptr =
        std::make_unique<CySolverResult>(integration_method);

    // Run
    baseline_cysolve_ivp_noreturn(
        solution_uptr.get(),
        diffeq_ptr,
        t_start,
        t_end,
        y0_vec,
        expected_size,
        num_extra,
        args_vec,
        max_num_steps,
        max_ram_MB,
        capture_dense_output,
        t_eval_vec,
        pre_eval_func,
        rtols,
        atols,
        max_step_size,
        first_step_size
    );

    // Return the results
    return std::move(solution_uptr);
}
