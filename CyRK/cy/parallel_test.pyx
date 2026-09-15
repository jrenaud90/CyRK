# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
""" Checks that CySolver can be driven from several C++ threads at once.

Every job solves the same Lotka-Volterra problem through `c_cysolve_batch`, and the results are compared with the
first job's, so any interference between threads shows up as a mismatch. Each runner returns its wall-clock time so
the tests can compare thread counts. This module is also the worked example for `CyRK.cy.parallel`.
"""
import time

from libcpp.vector cimport vector
from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from CyRK.cy.cysolver_api cimport CySolverResult, CySolveOutput, ODEMethod
from CyRK.cy.cysolver_test cimport lorenz_extraoutput_diffeq, lotkavolterra_diffeq
from CyRK.cy.parallel cimport c_CySolveJob, c_cysolve_batch

cdef size_t NUM_RUNS = 50
cdef double T_START = 0.0
# Integrate for a long time so that each job is slow enough to benefit from running in parallel.
cdef double T_END = 20_000.0
cdef double RTOL = 1.0e-5
cdef double ATOL = 1.0e-8


cdef extern from *:
    """
    #include <atomic>
    #include <cstring>
    #include <stdexcept>
    #include <thread>
    #include <vector>

    #include "cysolution.hpp"

    /* Reads one solution's dense output from `num_threads` threads at once and counts the reads that differ from a
       serial read of the same time. Every thread calls `call` at each time in `t_ptr`, `num_repeats` times over;
       even and odd threads walk the times in opposite orders so their calls overlap at different times. */
    inline size_t c_count_concurrent_dense_mismatches(
            CySolverResult* solution_ptr,
            const double* t_ptr,
            size_t len_t,
            size_t num_threads,
            size_t num_repeats)
    {
        const size_t num_dy = solution_ptr->num_dy;
        std::vector<double> reference(len_t * num_dy);
        for (size_t k = 0; k < len_t; k++)
        {
            solution_ptr->call(t_ptr[k], &reference[k * num_dy]);
        }

        std::atomic<bool> start(false);
        std::vector<size_t> mismatches(num_threads, 0);
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        try
        {
            for (size_t thread_i = 0; thread_i < num_threads; thread_i++)
            {
                workers.emplace_back([&, thread_i]()
                {
                    std::vector<double> y_interp(num_dy);
                    while (!start.load())
                    {
                        std::this_thread::yield();
                    }
                    for (size_t repeat = 0; repeat < num_repeats; repeat++)
                    {
                        for (size_t step = 0; step < len_t; step++)
                        {
                            const size_t k = (thread_i % 2 == 0) ? step : len_t - 1 - step;
                            solution_ptr->call(t_ptr[k], y_interp.data());
                            if (std::memcmp(y_interp.data(), &reference[k * num_dy], sizeof(double) * num_dy) != 0)
                            {
                                mismatches[thread_i]++;
                            }
                        }
                    }
                });
            }
        }
        catch (...)
        {
            // A thread could not be started: release and join the ones that were before passing the error on.
            start.store(true);
            for (std::thread& worker : workers)
            {
                worker.join();
            }
            throw;
        }
        start.store(true);
        for (std::thread& worker : workers)
        {
            worker.join();
        }

        size_t total = 0;
        for (const size_t count : mismatches)
        {
            total += count;
        }
        return total;
    }
    """
    size_t c_count_concurrent_dense_mismatches(
        CySolverResult* solution_ptr,
        const double* t_ptr,
        size_t len_t,
        size_t num_threads,
        size_t num_repeats) except + nogil


cdef void set_lotkavolterra_args(vector[char]& args_vec) noexcept nogil:
    """ Store the four Lotka-Volterra coefficients as doubles in the solver's byte-vector argument format. """
    args_vec.resize(4 * sizeof(double))
    cdef double* args_dbl_ptr = <double*>args_vec.data()
    args_dbl_ptr[0] = 1.5
    args_dbl_ptr[1] = 1.0
    args_dbl_ptr[2] = 3.0
    args_dbl_ptr[3] = 1.0


def _run_batch(size_t num_threads, bint share_args, size_t num_runs = NUM_RUNS, double t_end = T_END):
    """ Solve `num_runs` identical problems over `num_threads` threads, check the results, and return the time in ms.

    With `share_args` every job reads one shared argument vector; otherwise each job gets its own copy.
    """
    if num_runs == 0:
        raise ValueError("num_runs must be at least 1.")
    cdef size_t i
    cdef vector[double] y0_vec = vector[double](2)
    y0_vec[0] = 10.0
    y0_vec[1] = 5.0

    cdef vector[vector[char]] args_vecs = vector[vector[char]](1 if share_args else num_runs)
    for i in range(args_vecs.size()):
        set_lotkavolterra_args(args_vecs[i])

    # Each job owns its result; everything else is read only during the solve and may be shared.
    cdef vector[CySolveOutput] results = vector[CySolveOutput](num_runs)
    cdef vector[c_CySolveJob] jobs = vector[c_CySolveJob]()
    jobs.reserve(num_runs)
    cdef c_CySolveJob job
    for i in range(num_runs):
        results[i] = move(make_unique[CySolverResult](ODEMethod.RK45))
        job = c_CySolveJob()
        job.solution_ptr = results[i].get()
        job.diffeq_ptr   = lotkavolterra_diffeq
        job.t_start      = T_START
        job.t_end        = t_end
        job.y0_vec_ptr   = &y0_vec
        job.args_vec_ptr = &args_vecs[0 if share_args else i]
        job.rtol         = RTOL
        job.atol         = ATOL
        jobs.push_back(job)

    start_time = time.perf_counter()
    with nogil:
        c_cysolve_batch(jobs, num_threads)
    cdef double loop_time_ms = 1000.0 * (time.perf_counter() - start_time)

    # Every job must have succeeded and must match the first one exactly.
    cdef CySolverResult* reference_ptr = results[0].get()
    if not reference_ptr.success:
        raise AssertionError(f"Reference job failed: {reference_ptr.message.decode()}")
    if reference_ptr.size == 0:
        raise AssertionError("Reference job produced no steps.")

    cdef CySolverResult* check_ptr
    cdef size_t j
    for i in range(1, num_runs):
        check_ptr = results[i].get()
        if not check_ptr.success:
            raise AssertionError(f"Job {i} failed: {check_ptr.message.decode()}")
        if check_ptr.size != reference_ptr.size:
            raise AssertionError(f"Job {i} took {check_ptr.size} steps; the reference took {reference_ptr.size}.")
        for j in range(reference_ptr.size):
            if (check_ptr.time_domain_vec[j] != reference_ptr.time_domain_vec[j]
                    or check_ptr.solution[2 * j] != reference_ptr.solution[2 * j]
                    or check_ptr.solution[2 * j + 1] != reference_ptr.solution[2 * j + 1]):
                raise AssertionError(f"Job {i} differs from the reference at step {j}.")

    return loop_time_ms


def run_parallel_test(size_t num_threads = 2):
    """ Solve a batch where each job has its own argument vector, spread over `num_threads` C++ threads. """
    loop_time_ms = _run_batch(num_threads, False)
    print(f"CyRK parallel test `run_parallel_test` finished in {loop_time_ms:0.1f} ms on {num_threads} thread(s) "
          "(if you are seeing this then the checks passed too).")
    print(f"\tAvg time: {loop_time_ms / NUM_RUNS:0.1f} ms/job.")
    return loop_time_ms


def run_parallel_common_args_test(size_t num_threads = 2):
    """ Solve a batch where every job reads the same argument vector, spread over `num_threads` C++ threads. """
    loop_time_ms = _run_batch(num_threads, True)
    print(f"CyRK parallel test `run_parallel_common_args_test` finished in {loop_time_ms:0.1f} ms on {num_threads} "
          "thread(s) (if you are seeing this then the checks passed too).")
    print(f"\tAvg time: {loop_time_ms / NUM_RUNS:0.1f} ms/job.")
    return loop_time_ms


def run_dense_output_threads_test(size_t num_threads = 4, size_t num_repeats = 100):
    """ Solve a Lorenz problem with extra outputs once, then read its dense output from `num_threads` C++ threads.

    Every thread compares each read, extra outputs included, with a serial read at the same time. A dense read only
    reads the shared solution, so no read may differ. Returns the number of reads that did.
    """
    if num_threads == 0 or num_repeats == 0:
        raise ValueError("num_threads and num_repeats must each be at least 1.")
    cdef size_t i
    cdef vector[double] y0_vec = vector[double](3)
    y0_vec[0] = 1.0
    y0_vec[1] = 0.0
    y0_vec[2] = 0.0

    # The Lorenz coefficients (sigma, rho, beta) as doubles in the solver's byte-vector argument format.
    cdef vector[char] args_vec = vector[char](3 * sizeof(double))
    cdef double* args_dbl_ptr = <double*>args_vec.data()
    args_dbl_ptr[0] = 10.0
    args_dbl_ptr[1] = 28.0
    args_dbl_ptr[2] = 8.0 / 3.0

    cdef vector[CySolveOutput] results = vector[CySolveOutput](1)
    results[0] = move(make_unique[CySolverResult](ODEMethod.DOP853))
    cdef vector[c_CySolveJob] jobs = vector[c_CySolveJob](1)
    jobs[0].solution_ptr         = results[0].get()
    jobs[0].diffeq_ptr           = lorenz_extraoutput_diffeq
    jobs[0].t_start              = 0.0
    jobs[0].t_end                = 10.0
    jobs[0].y0_vec_ptr           = &y0_vec
    jobs[0].args_vec_ptr         = &args_vec
    jobs[0].num_extra            = 3
    jobs[0].capture_dense_output = True
    jobs[0].rtol                 = RTOL
    jobs[0].atol                 = ATOL
    with nogil:
        c_cysolve_batch(jobs, 1)

    cdef CySolverResult* solution_ptr = results[0].get()
    if not solution_ptr.success:
        raise AssertionError(f"The Lorenz solve failed: {solution_ptr.message.decode()}")

    # Read times spread across the solution, stopping short of its end.
    cdef size_t len_t = 1000
    cdef vector[double] t_vec = vector[double](len_t)
    for i in range(len_t):
        t_vec[i] = 10.0 * (<double>i) / (<double>len_t)

    cdef size_t num_mismatches
    with nogil:
        num_mismatches = c_count_concurrent_dense_mismatches(
            solution_ptr, t_vec.data(), len_t, num_threads, num_repeats)
    return num_mismatches


def run_parallel_benchmark(size_t num_threads, size_t num_runs, double t_end):
    """ Time `num_runs` identical Lotka-Volterra solves to `t_end` on `num_threads` threads; returns the ms taken.

    Use it to find where threading starts to pay off on your machine. The per-job time grows with `t_end` (the
    solution is periodic, so the step count is proportional to it) on top of the solver's fixed setup cost.
    """
    return _run_batch(num_threads, True, num_runs, t_end)
