#include "cysolve.hpp"


std::shared_ptr<CySolverResult> baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const unsigned int num_y,
    const unsigned int method,
    // General optional arguments
    const size_t expected_size,
    const unsigned int num_extra,
    const double* args_ptr,
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
)
{
    // State parameters
    bool error = false;

    // Parse input
    const double t_start       = t_span_ptr[0];
    const double t_end         = t_span_ptr[1];
    const bool direction_flag  = t_start <= t_end ? true : false;
    const bool t_eval_provided = t_eval ? true : false;

    // Get new expected size
    size_t expected_size_touse = expected_size;
    if (expected_size_touse == 0)
    {
        double min_rtol = INF;
        if (rtols_ptr)
        {
            // rtol for each y
            for (unsigned int y_i = 0; y_i < num_y; y_i++)
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

    // Build storage class
    std::shared_ptr<CySolverResult> solution_ptr =
        std::make_shared<CySolverResult>(
            num_y,
            num_extra,
            expected_size_touse,
            t_end,
            direction_flag,
            dense_output,
            t_eval_provided);

    // Build solver class
    CySolverBase* solver;

    switch (method)
    {
    case 0:
        // RK23
        solver = new RK23(
            // Common Inputs
            diffeq_ptr, solution_ptr, t_start, t_end, y0_ptr, num_y, num_extra, args_ptr, max_num_steps, max_ram_MB,
            dense_output, t_eval, len_t_eval,
            // RK Inputs
            rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
        );
        break;
    case 1:
        // RK45
        solver = new RK45(
            // Common Inputs
            diffeq_ptr, solution_ptr, t_start, t_end, y0_ptr, num_y, num_extra, args_ptr, max_num_steps, max_ram_MB,
            dense_output, t_eval, len_t_eval,
            // RK Inputs
            rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
        );
        break;
    case 2:
        // DOP853
        solver = new DOP853(
            // Common Inputs
            diffeq_ptr, solution_ptr, t_start, t_end, y0_ptr, num_y, num_extra, args_ptr, max_num_steps, max_ram_MB,
            dense_output, t_eval, len_t_eval,
            // RK Inputs
            rtol, atol, rtols_ptr, atols_ptr, max_step_size, first_step_size
        );
        break;
    [[unlikely]] default:
        error = true;
        solution_ptr->success = false;
        solution_ptr->error_code = -3;
        solution_ptr->update_message("Model Error: Not implemented or unknown CySolver model requested.\n");
        return solution_ptr;
        break;
    }

    // Call reset on solver
    solver->reset();

    // Run integrator
    solver->solve();

    // Finalize solution storage
    solution_ptr->finalize();

    // Delete solver instance
    delete solver;

    // Return the results
    return solution_ptr;
}


/* Pure Python hook solvers and helpers */
PySolver::PySolver() { }
PySolver::~PySolver()
{
    if (this->solver)
    {
        delete this->solver;
    }
}
PySolver::PySolver(
    unsigned int integration_method,
    // Cython class instance used for pyhook
    PyObject* cython_extension_class_instance,
    DiffeqMethod cython_extension_class_diffeq_method,
    // Regular integrator inputs
    std::shared_ptr<CySolverResult> solution_ptr,
    const double t_start,
    const double t_end,
    const double* y0_ptr,
    const unsigned int num_y,
    // General optional arguments
    const unsigned int num_extra,
    const double* args_ptr,
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
        integration_method(integration_method)

{   
    // Save reference to solution pointer
    this->solution_ptr = solution_ptr;

    // We need to pass a fake diffeq pointer (diffeq ptr is unused in python-based solver)
    DiffeqFuncType diffeq_ptr = nullptr;

    // Build the solver class. This must be heap allocated to take advantage of polymorphism.
    switch (this->integration_method)
    {
    case 0:
        // RK23
        this->solver = new RK23(
            diffeq_ptr,
            solution_ptr,
            t_start,
            t_end,
            y0_ptr,
            num_y,
            // General optional arguments
            num_extra,
            args_ptr,
            max_num_steps,
            max_ram_MB,
            dense_output,
            t_eval,
            len_t_eval,
            // rk optional arguments
            rtol,
            atol,
            rtols_ptr,
            atols_ptr,
            max_step_size,
            first_step_size);
        break;
    case 1:
        // RK45
        this->solver = new RK45(
            diffeq_ptr,
            solution_ptr,
            t_start,
            t_end,
            y0_ptr,
            num_y,
            // General optional arguments
            num_extra,
            args_ptr,
            max_num_steps,
            max_ram_MB,
            dense_output,
            t_eval,
            len_t_eval,
            // rk optional arguments
            rtol,
            atol,
            rtols_ptr,
            atols_ptr,
            max_step_size,
            first_step_size);
        break;
    case 2:
        // DOP853
        this->solver = new DOP853(
            diffeq_ptr,
            solution_ptr,
            t_start,
            t_end,
            y0_ptr,
            num_y,
            // General optional arguments
            num_extra,
            args_ptr,
            max_num_steps,
            max_ram_MB,
            dense_output,
            t_eval,
            len_t_eval,
            // rk optional arguments
            rtol,
            atol,
            rtols_ptr,
            atols_ptr,
            max_step_size,
            first_step_size);
        break;
    [[unlikely]] default:
        this->status = -1;
        break;
    }

    if (this->solver) [[likely]]
    {
        // Add in python hooks
        this->solver->set_cython_extension_instance(cython_extension_class_instance, cython_extension_class_diffeq_method);

        // Install solver's state pointers
        this->state_pointers = PySolverStatePointers(
            this->solver->dy_now_ptr,
            this->solver->t_now_ptr,
            this->solver->y_now_ptr);
    }
};


PySolverStatePointers PySolver::get_state_pointers() const
{
    /*
    The differential equation is called from the python side but it must use and update state pointers in the
    CySolver class instance. This function returns all required pointers so Python can properly use and update them.
    */
    return this->state_pointers;
}


void PySolver::solve()
{
    // Run integrator
    if (this->solver) [[likely]]
    {
        this->solver->solve();
    }

    // Finalize solution storage
    this->solution_ptr->finalize();
}
