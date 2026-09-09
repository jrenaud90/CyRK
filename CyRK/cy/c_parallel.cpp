#include "c_parallel.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#include "cysolve.hpp"

namespace {

/* Solves one job in the current thread. Errors are recorded on the job's result because an exception escaping a
   worker thread would terminate the whole process. */
void solve_job(c_CySolveJob& job)
{
    // The solver copies each vector it is handed, so stand-ins for inputs the job did not provide can live here.
    std::vector<double> empty_dbl_vec;
    std::vector<char> empty_char_vec;
    std::vector<Event> empty_event_vec;
    std::vector<double> rtols_vec(1, job.rtol);
    std::vector<double> atols_vec(1, job.atol);

    try {
        baseline_cysolve_ivp_noreturn(
            job.solution_ptr,
            job.diffeq_ptr,
            job.t_start,
            job.t_end,
            *job.y0_vec_ptr,
            job.expected_size,
            job.num_extra,
            job.args_vec_ptr ? *job.args_vec_ptr : empty_char_vec,
            job.max_num_steps,
            job.max_ram_MB,
            job.capture_dense_output,
            job.t_eval_vec_ptr ? *job.t_eval_vec_ptr : empty_dbl_vec,
            job.pre_eval_func,
            job.events_vec_ptr ? *job.events_vec_ptr : empty_event_vec,
            job.rtols_vec_ptr ? *job.rtols_vec_ptr : rtols_vec,
            job.atols_vec_ptr ? *job.atols_vec_ptr : atols_vec,
            job.max_step_size,
            job.first_step_size,
            job.force_retain_solver,
            job.jac_ptr);
    }
    catch (const std::exception& error) {
        job.solution_ptr->update_status(CyrkErrorCodes::GENERAL_ERROR);
        job.solution_ptr->message = std::string("Exception raised while solving a batch job: ") + error.what();
    }
    catch (...) {
        job.solution_ptr->update_status(CyrkErrorCodes::GENERAL_ERROR);
        job.solution_ptr->message = "Unknown exception raised while solving a batch job.";
    }
}

}  // namespace

size_t c_cysolve_batch(std::vector<c_CySolveJob>& jobs, size_t num_threads)
{
    for (const c_CySolveJob& job : jobs) {
        if (!job.solution_ptr || !job.diffeq_ptr || !job.y0_vec_ptr) {
            throw std::invalid_argument(
                "c_cysolve_batch: every job needs a solution_ptr, a diffeq_ptr, and a y0_vec_ptr.");
        }
    }

    const size_t num_jobs     = jobs.size();
    const size_t threads_used = std::max<size_t>(1, std::min(num_threads, num_jobs));
    if (threads_used == 1) {
        for (c_CySolveJob& job : jobs) {
            solve_job(job);
        }
        return 1;
    }

    // Hand each thread a contiguous block of jobs; the first `leftover` threads take one extra.
    const size_t jobs_per_thread = num_jobs / threads_used;
    const size_t leftover        = num_jobs % threads_used;
    std::vector<std::thread> workers;
    workers.reserve(threads_used);
    size_t first_job = 0;
    try {
        for (size_t thread_i = 0; thread_i < threads_used; thread_i++) {
            const size_t last_job = first_job + jobs_per_thread + (thread_i < leftover ? 1 : 0);
            workers.emplace_back(
                [&jobs, first_job, last_job]() {
                    for (size_t job_i = first_job; job_i < last_job; job_i++) {
                        solve_job(jobs[job_i]);
                    }
                });
            first_job = last_job;
        }
    }
    catch (...) {
        // A thread could not be started; wait for the ones that were before passing the error on.
        for (std::thread& worker : workers) {
            worker.join();
        }
        throw;
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    return threads_used;
}
