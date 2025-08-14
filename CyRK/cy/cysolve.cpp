#include <stdexcept>

#include "cysolve.hpp"
#include "cysolver.hpp"

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
        std::vector<double>& rtols,
        std::vector<double>& atols,
        double max_step_size,
        double first_step_size
        )
{
    // For now we are only ever using RK methods so it is safe to assume RK Config.
    RKConfig* rk_config_ptr = static_cast<RKConfig*>(solution_ptr->config_uptr.get());
    // Change the configurations of the solver.
    rk_config_ptr->update_properties(
        diffeq_ptr,
        t_start,
        t_end,
        y0_vec,
        args_vec,
        t_eval_vec,
        num_extra,
        expected_size,
        max_num_steps,
        max_ram_MB,
        pre_eval_func,
        capture_dense_output,
        true, // force_retain_solver; not currently in use.
        rtols,
        atols,
        max_step_size,
        first_step_size
    );

    // Initialize the solution and solver given the new configurations
    solution_ptr->setup(nullptr);

    // Run integrator
    solution_ptr->solve();
}

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
        std::vector<double>& rtols,
        std::vector<double>& atols,
        double max_step_size,
        double first_step_size
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
