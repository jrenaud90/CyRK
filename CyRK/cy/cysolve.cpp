#include "cysolve.hpp"


template <typename T>
void method_solve(
        DiffeqFuncType diffeq_ptr,
        std::shared_ptr<CySolverResult> solution_ptr,
        const double t_start,
        const double t_end,
        double* y0_ptr,
        size_t num_y,
        // General optional arguments
        bool capture_extra,
        size_t num_extra,
        double* args_ptr,
        // rk optional arguments
        size_t max_num_steps,
        size_t max_ram_MB,
        double rtol,
        double atol,
        double* rtols_ptr,
        double* atols_ptr,
        double max_step_size,
        double first_step_size)
{
    // Construct solver based on type
    T solver = T(
        // Common Inputs
        diffeq_ptr, solution_ptr, t_start, t_end, y0_ptr, num_y,
        // RK Inputs
        capture_extra, num_extra, args_ptr, max_num_steps, max_ram_MB, rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
    );

    // Run integrator
    while (solver.check_status())
    {
        solver.take_step();
    }
}


std::shared_ptr<CySolverResult> cysolve_ivp(
        DiffeqFuncType diffeq_ptr,
        double* t_span_ptr,
        double* y0_ptr,
        size_t num_y,
        int method,
        // General optional arguments
        size_t expected_size,
        bool capture_extra,
        size_t num_extra,
        double* args_ptr,
        // rk optional arguments
        size_t max_num_steps,
        size_t max_ram_MB,
        double rtol,
        double atol,
        double* rtols_ptr,
        double* atols_ptr,
        double max_step_size,
        double first_step_size
        )
{
    // State parameters
    bool error = false;

    // Parse input
    const double t_start = t_span_ptr[0];
    const double t_end = t_span_ptr[1];
    double min_rtol = INF;
    double rtol_tmp;

    // Get new expected size
    if (expected_size == 0)
    {
        if (rtols_ptr)
        {
            // rtol for each y
            for (size_t y_i = 0; y_i < num_y; y_i++)
            {
                rtol_tmp = rtols_ptr[y_i];
                if (rtol_tmp < EPS_100)
                {
                    rtol_tmp = EPS_100;
                }
                min_rtol = std::fmin(min_rtol, rtol_tmp);
            }
        }
        else {
            // only one rtol
            rtol_tmp = rtol;
            if (rtol_tmp < EPS_100)
            {
                rtol_tmp = EPS_100;
            }
            min_rtol = rtol_tmp;
        }
        expected_size = find_expected_size(num_y, num_extra, std::fabs(t_end - t_start), min_rtol, capture_extra);
    }

    // Build classes
    std::shared_ptr<CySolverResult> solution_ptr = std::make_shared<CySolverResult>(num_y, num_extra, expected_size);

    switch (method)
    {
        case 0:
            // RK23
            method_solve<RK23>(
                // Common Inputs
                diffeq_ptr, solution_ptr, t_start, t_end, y0_ptr, num_y,
                // RK Inputs
                capture_extra, num_extra, args_ptr, max_num_steps, max_ram_MB, rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
            );
            break;
        case 1:
            // RK45
            method_solve<RK45>(
                // Common Inputs
                diffeq_ptr, solution_ptr, t_start, t_end, y0_ptr, num_y,
                // RK Inputs
                capture_extra, num_extra, args_ptr, max_num_steps, max_ram_MB, rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
            );
            break;
        case 2:
            // DOP853
            method_solve<DOP853>(
                // Common Inputs
                diffeq_ptr, solution_ptr, t_start, t_end, y0_ptr, num_y,
                // RK Inputs
                capture_extra, num_extra, args_ptr, max_num_steps, max_ram_MB, rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
            );
            break;
        default:
            error = true;
            solution_ptr->success = false;
            solution_ptr->error_code = -3;
            solution_ptr->update_message("Model Error: Not implemented or unknown CySolver model requested.\n");
            break;
    }

    return solution_ptr;
}
