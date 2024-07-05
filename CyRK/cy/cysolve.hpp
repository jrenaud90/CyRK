#pragma once

#include <memory>

#include "common.hpp"
#include "rk.hpp"
#include "cysolver.hpp"

/* Pure C++ / Cython solvers and helpers */
template <typename IntegratorType>
void find_cysolver_and_solve(
    DiffeqFuncType diffeq_ptr,
    std::shared_ptr<CySolverResult> solution_ptr,
    const double t_start,
    const double t_end,
    const double* y0_ptr,
    const unsigned int num_y,
    // General optional arguments
    const unsigned int num_extra,
    const double* args_ptr,
    // rk optional arguments
    const size_t max_num_steps,
    const size_t max_ram_MB,
    const double rtol,
    const double atol,
    const double* rtols_ptr,
    const double* atols_ptr,
    const double max_step_size,
    const double first_step_size
);

std::shared_ptr<CySolverResult> cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const unsigned int num_y,
    const unsigned int method,
    // General optional arguments
    const size_t expected_size = 0,
    const unsigned int num_extra = 0,
    const double* args_ptr = nullptr,
    // rk optional arguments
    const size_t max_num_steps = 0,
    const size_t max_ram_MB = 2000,
    const double rtol = 1.0e-3,
    const double atol = 1.0e-6,
    const double* rtols_ptr = nullptr,
    const double* atols_ptr = nullptr,
    const double max_step_size = MAX_STEP,
    const double first_step_size = 0.0
);


/* Pure Python hook solvers and helpers */
struct PySolverStatePointers
{
    double* dy_now_ptr;
    double* t_now_ptr;
    double* y_now_ptr;
    PySolverStatePointers() :
        dy_now_ptr(nullptr), t_now_ptr(nullptr), y_now_ptr(nullptr) {};
    PySolverStatePointers(double* dy_now_ptr, double* t_now_ptr, double* y_now_ptr) :
        dy_now_ptr(dy_now_ptr), t_now_ptr(t_now_ptr), y_now_ptr(y_now_ptr) {};
};

class PySolver
{
public:
    CySolverBase* solver;
    std::shared_ptr<CySolverResult> solution_ptr = std::make_shared<CySolverResult>();
    unsigned int integration_method = 1;
    PySolverStatePointers state_pointers = PySolverStatePointers();

public:
    PySolver();
    ~PySolver();
    PySolver(
        unsigned int integration_method,
        // Cython class instance used for pyhook
        PyObject* cython_extension_class_instance,
        // Regular integrator inputs
        std::shared_ptr<CySolverResult> solution_ptr,
        const double t_start,
        const double t_end,
        const double* y0_ptr,
        const unsigned int num_y,
        // General optional arguments
        const unsigned int num_extra,
        const double* args_ptr,
        // rk optional arguments
        const size_t max_num_steps,
        const size_t max_ram_MB,
        const double rtol,
        const double atol,
        const double* rtols_ptr,
        const double* atols_ptr,
        const double max_step_size,
        const double first_step_size
    );
    PySolverStatePointers get_state_pointers() const;
    void solve();
};