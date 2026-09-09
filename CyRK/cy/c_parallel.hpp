#pragma once
/* Runs a batch of independent CySolver integrations across C++ threads.

   The solver copies every input vector it is handed, so jobs may share their initial
   condition, argument, tolerance, t_eval, and event vectors. Each job must own is its CySolverResult:
   it is the only object a worker thread writes to. */

#include <cstddef>
#include <vector>

#include "c_common.hpp"
#include "c_events.hpp"
#include "cysolution.hpp"

/* One integration problem for `c_cysolve_batch`. The defaults match the Cython wrapper `cysolve_ivp_noreturn`;
   a null optional pointer means "not provided". */
struct c_CySolveJob {
    // Required
    CySolverResult* solution_ptr = nullptr;   // Receives the solution; must be unique to this job.
    DiffeqFuncType diffeq_ptr    = nullptr;
    double t_start = 0.0;
    double t_end   = 0.0;
    std::vector<double>* y0_vec_ptr = nullptr;

    // Optional
    std::vector<char>* args_vec_ptr = nullptr;
    double rtol = 1.0e-3;
    double atol = 1.0e-6;
    std::vector<double>* rtols_vec_ptr = nullptr;   // Per-variable tolerances; override `rtol` / `atol` when set.
    std::vector<double>* atols_vec_ptr = nullptr;
    size_t expected_size = 0;
    size_t num_extra     = 0;
    size_t max_num_steps = 0;
    size_t max_ram_MB    = 2000;
    bool capture_dense_output = false;
    std::vector<double>* t_eval_vec_ptr = nullptr;
    PreEvalFunc pre_eval_func = nullptr;
    std::vector<Event>* events_vec_ptr = nullptr;
    double max_step_size   = MAX_STEP;
    double first_step_size = 0.0;
    bool force_retain_solver = true;
    JacobianFuncType jac_ptr = nullptr;
};

/* Solves every job in `jobs`, spread over `num_threads` C++ threads as contiguous blocks (like a static OpenMP
   schedule). A `num_threads` of 0 or 1 solves them in the calling thread; more threads than jobs are never started.
   Throws std::invalid_argument, before any job starts, if a job is missing a required pointer. A failure inside a
   job is recorded on that job's CySolverResult rather than raised. Returns the number of threads used. */
size_t c_cysolve_batch(std::vector<c_CySolveJob>& jobs, size_t num_threads);
