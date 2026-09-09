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
from CyRK.cy.cysolver_test cimport lotkavolterra_diffeq
from CyRK.cy.parallel cimport c_CySolveJob, c_cysolve_batch

cdef size_t NUM_RUNS = 50
cdef double T_START = 0.0
# Integrate for a long time so that each job is slow enough to benefit from running in parallel.
cdef double T_END = 20_000.0
cdef double RTOL = 1.0e-5
cdef double ATOL = 1.0e-8


cdef void set_lotkavolterra_args(vector[char]& args_vec) noexcept nogil:
    """ Store the four Lotka-Volterra coefficients as doubles in the solver's byte-vector argument format. """
    args_vec.resize(4 * sizeof(double))
    cdef double* args_dbl_ptr = <double*>args_vec.data()
    args_dbl_ptr[0] = 1.5
    args_dbl_ptr[1] = 1.0
    args_dbl_ptr[2] = 3.0
    args_dbl_ptr[3] = 1.0


def _run_batch(size_t num_threads, bint share_args):
    """ Solve NUM_RUNS identical problems over `num_threads` threads, check the results, and return the time in ms.

    With `share_args` every job reads one shared argument vector; otherwise each job gets its own copy.
    """
    cdef size_t i
    cdef vector[double] y0_vec = vector[double](2)
    y0_vec[0] = 10.0
    y0_vec[1] = 5.0

    cdef vector[vector[char]] args_vecs = vector[vector[char]](1 if share_args else NUM_RUNS)
    for i in range(args_vecs.size()):
        set_lotkavolterra_args(args_vecs[i])

    # Each job owns its result; everything else is read only during the solve and may be shared.
    cdef vector[CySolveOutput] results = vector[CySolveOutput](NUM_RUNS)
    cdef vector[c_CySolveJob] jobs = vector[c_CySolveJob]()
    jobs.reserve(NUM_RUNS)
    cdef c_CySolveJob job
    for i in range(NUM_RUNS):
        results[i] = move(make_unique[CySolverResult](ODEMethod.RK45))
        job = c_CySolveJob()
        job.solution_ptr = results[i].get()
        job.diffeq_ptr   = lotkavolterra_diffeq
        job.t_start      = T_START
        job.t_end        = T_END
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
    for i in range(1, NUM_RUNS):
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
