#include "cysolve.hpp"
#include <exception>


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
        const size_t size_of_args,
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
    )
{
    // Parse input
    const double t_start       = t_span_ptr[0];
    const double t_end         = t_span_ptr[1];
    const bool direction_flag  = t_start <= t_end ? true : false;
    const bool forward = direction_flag == true;
    const bool t_eval_provided = t_eval ? true : false;

    // Get new expected size
    size_t expected_size_touse = expected_size;
    if (expected_size_touse == 0)
    {
        double min_rtol = INF;
        if (rtols_ptr)
        {
            // rtol for each y
            for (size_t y_i = 0; y_i < num_y; y_i++)
            {
                double rtol_tmp = rtols_ptr[y_i];
                if (rtol_tmp < EPS_100)
                {
                    rtol_tmp = EPS_100;
                }
                min_rtol = std::fmin(min_rtol, rtol_tmp);
            }
        }
        else {
            // only one rtol
            double rtol_tmp = rtol;
            if (rtol_tmp < EPS_100)
            {
                rtol_tmp = EPS_100;
            }
            min_rtol = rtol_tmp;
        }
        expected_size_touse = find_expected_size(num_y, num_extra, std::fabs(t_end - t_start), min_rtol);
    }

    // Set the expected size of the arrays
    solution_sptr->set_expected_size(expected_size_touse);

    // Setup solver class
    solution_sptr->build_solver(
        diffeq_ptr,
        t_start,
        t_end,
        y0_ptr,
        method,
        // General optional arguments
        expected_size,
        args_ptr,
        size_of_args,
        max_num_steps,
        max_ram_MB,
        t_eval,
        len_t_eval,
        pre_eval_func,
        // rk optional arguments
        rtol,
        atol,
        rtols_ptr,
        atols_ptr,
        max_step_size,
        first_step_size
    );
    // Run integrator
    solution_sptr->solve();
}

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
        const size_t size_of_args,
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
    )
{
    const double t_start       = t_span_ptr[0];
    const double t_end         = t_span_ptr[1];
    const bool direction_flag  = t_start <= t_end ? true : false;
    const bool t_eval_provided = t_eval ? true : false;

    // Build storage class
    std::shared_ptr<CySolverResult> solution_sptr =
        std::make_shared<CySolverResult>(
            num_y,
            num_extra,
            expected_size,
            t_end,
            direction_flag,
            dense_output,
            t_eval_provided);

    // Run
    baseline_cysolve_ivp_noreturn(
        solution_sptr,
        diffeq_ptr,
        t_span_ptr,
        y0_ptr,
        num_y,
        method,
        // General optional arguments
        expected_size,
        num_extra,
        args_ptr,
        size_of_args,
        max_num_steps,
        max_ram_MB,
        dense_output,
        t_eval,
        len_t_eval,
        pre_eval_func,
        // rk optional arguments
        rtol,
        atol,
        rtols_ptr,
        atols_ptr,
        max_step_size,
        first_step_size
    );

    // Return the results
    return solution_sptr;
}


/* Pure Python hook solvers and helpers */
PySolver::PySolver()
{

}

PySolver::~PySolver()
{

}


PySolver::PySolver(
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
        const double first_step_size) :
            status(0),
            integration_method(integration_method),
            solution_sptr(solution_sptr)

{ 
    // We need to pass a fake diffeq pointer (diffeq ptr is unused in python-based solver)
    DiffeqFuncType diffeq_ptr = nullptr;

    // We also need to pass a fake pre-eval function
    PreEvalFunc pre_eval_func = nullptr;

    // Args are handled by the python class too.
    const char* args_ptr      = nullptr;
    const size_t size_of_args = 0;

    // Build the solver class. This must be heap allocated to take advantage of polymorphism.
    // Setup solver class
    if (this->solution_sptr) [[likely]]
    {
        this->solution_sptr->build_solver(
            diffeq_ptr,  // not used when using pysolver
            t_start,
            t_end,
            y0_ptr,
            integration_method,
            // General optional arguments
            expected_size,
            args_ptr,
            size_of_args,
            max_num_steps,
            max_ram_MB,
            t_eval,
            len_t_eval,
            pre_eval_func,  // not used when using pysolver
            // rk optional arguments
            rtol,
            atol,
            rtols_ptr,
            atols_ptr,
            max_step_size,
            first_step_size
        );

        // Add in python hooks
        this->solution_sptr->solver_uptr->set_cython_extension_instance(cython_extension_class_instance, cython_extension_class_diffeq_method);
    }
    else
    {
        throw std::exception(); // "Solution storage not created. Perhaps memory issue or bad alloc."
    }
};


void PySolver::solve()
{
    // Run integrator
    if (this->solution_sptr)
    {
        // Reset solver to t0
        this->solution_sptr->solve();
    }
    else
    {
        throw std::exception(); // Solution storage pointer no longer valid
    }
}
