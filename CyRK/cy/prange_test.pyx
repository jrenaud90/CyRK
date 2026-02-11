# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from cython.parallel cimport prange

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move

from CyRK.cy.cysolver_api cimport cysolve_ivp_noreturn, CySolverResult, CySolveOutput, ODEMethod
from CyRK.cy.cysolver_test cimport lotkavolterra_diffeq

cdef extern from "<chrono>" namespace "std::chrono" nogil:
    cdef cppclass microseconds:
        long long count()
    
    # We need a generic duration type to handle the result of subtraction
    cdef cppclass duration "std::chrono::high_resolution_clock::duration":
        pass

    microseconds duration_cast_microseconds "std::chrono::duration_cast<std::chrono::microseconds>"(duration)

cdef extern from "<chrono>" namespace "std::chrono::high_resolution_clock" nogil:
    cdef cppclass time_point:
        duration operator-(time_point)
    
    time_point now()

def run_prange_test(int num_threads = 2):
    """Run a prange test where everything is built outside of the loop."""
    
    cdef int num_runs = 50
    cdef double t_start = 0.0
    # Pick a large time so each time step takes a significant amount of time so it can benefit from parallelize.
    cdef double t_end = 20_000.0
    
    cdef vector[vector[double]] y0_vecs = vector[vector[double]]()
    y0_vecs.resize(num_runs)
    cdef size_t i
    for i in range(num_runs):
        # Set initial conditions; all the same for these tests
        # 2 dependent variables
        y0_vecs[i].resize(2)
        y0_vecs[i][0] = 10.0
        y0_vecs[i][1] = 5.0
    
    cdef vector[vector[char]] arg_vecs = vector[vector[char]]()
    arg_vecs.resize(num_runs)
    cdef double* arg_dbl_ptr 
    for i in range(num_runs):
        # arg vector
        arg_vecs[i].resize(4 * 8)  # Number of double * size of double
        # Recast <char*> args to double pointer for storage of args
        arg_dbl_ptr = <double*>arg_vecs[i].data()
        arg_dbl_ptr[0] = 1.5
        arg_dbl_ptr[1] = 1.0
        arg_dbl_ptr[2] = 3.0
        arg_dbl_ptr[3] = 1.0

    cdef vector[CySolveOutput] results = vector[CySolveOutput]()
    results.resize(num_runs)
    for i in range(num_runs):
        results[i] = move(make_unique[CySolverResult](ODEMethod.RK45))

    cdef time_point start_time, end_time
    
    # Main Loop
    cdef Py_ssize_t p_i  # prange has to used a signed integer type
    if num_threads < 2:
        # Use regular loop
        start_time = now()
        for p_i in range(num_runs):

            cysolve_ivp_noreturn(
                results[p_i].get(), # Pass in CySolverResult pointer which is found with CySolveOutput::get()
                lotkavolterra_diffeq,
                t_start,
                t_end,
                y0_vecs[p_i],
                1.0e-5, # rtol
                1.0e-8, # atol
                arg_vecs[p_i]
            )
        end_time = now()
    else:
        # Use prange
        start_time = now()
        for p_i in prange(num_runs, nogil=True, num_threads=num_threads, schedule='static'):
            cysolve_ivp_noreturn(
                results[p_i].get(), # Pass in CySolverResult pointer which is found with CySolveOutput::get()
                lotkavolterra_diffeq,
                t_start,
                t_end,
                y0_vecs[p_i],
                1.0e-5, # rtol
                1.0e-8, # atol
                arg_vecs[p_i]
            )
        end_time = now()

    # Check results
    cdef CySolverResult* reference_result_ptr = results[0].get()
    # Check the reference first.
    assert reference_result_ptr.success
    assert reference_result_ptr.size > 0

    cdef CySolverResult* check_result_ptr
    cdef size_t j
    for i in range(num_runs):

        if i == 0:
            continue
        check_result_ptr = results[i].get()

        assert check_result_ptr.success
        assert check_result_ptr.size > 0
        assert reference_result_ptr.size == check_result_ptr.size

        for j in range(reference_result_ptr.size):
            assert check_result_ptr.time_domain_vec[j] == reference_result_ptr.time_domain_vec[j]
            assert check_result_ptr.solution[2 * j + 0] == reference_result_ptr.solution[2 * j + 0]
            assert check_result_ptr.solution[2 * j + 1] == reference_result_ptr.solution[2 * j + 1]


    cdef double loop_time_ms = <double>(<int>duration_cast_microseconds(end_time - start_time).count()) / 1000.0
    print(f"CyRK prange test `run_prange_outloop_test` finished in {loop_time_ms:0.1f} ms (If you are seeing this then tests passed too).")
    print(f"\tAvg time: {loop_time_ms/num_runs:0.1f}ms/loop.")

    return loop_time_ms


def run_prange_common_args_test(int num_threads = 2):
    """Run a prange test where args are common across all runs."""
    
    cdef int num_runs = 50
    cdef double t_start = 0.0
    # Pick a large time so each time step takes a significant amount of time so it can benefit from parallelize.
    cdef double t_end = 20_000.0
    
    cdef vector[vector[double]] y0_vecs = vector[vector[double]]()
    y0_vecs.resize(num_runs)
    cdef size_t i
    for i in range(num_runs):
        # Set initial conditions; all the same for these tests
        # 2 dependent variables
        y0_vecs[i].resize(2)
        y0_vecs[i][0] = 10.0
        y0_vecs[i][1] = 5.0
    
    # --- CHANGE: Single constant vector instead of vector[vector] ---
    cdef vector[char] constant_args = vector[char]()
    # arg vector
    constant_args.resize(4 * 8)  # Number of double * size of double
    
    # Recast <char*> args to double pointer for storage of args
    cdef double* arg_dbl_ptr = <double*> constant_args.data()
    arg_dbl_ptr[0] = 1.5
    arg_dbl_ptr[1] = 1.0
    arg_dbl_ptr[2] = 3.0
    arg_dbl_ptr[3] = 1.0

    cdef vector[CySolveOutput] results = vector[CySolveOutput]()
    results.resize(num_runs)
    for i in range(num_runs):
        results[i] = move(make_unique[CySolverResult](ODEMethod.RK45))

    cdef time_point start_time, end_time
    # Main Loop
    cdef Py_ssize_t p_i  # prange has to used a signed integer type
    if num_threads < 2:
        # Use regular loop
        start_time = now()
        for p_i in range(num_runs):

            cysolve_ivp_noreturn(
                results[p_i].get(), # Pass in CySolverResult pointer which is found with CySolveOutput::get()
                lotkavolterra_diffeq,
                t_start,
                t_end,
                y0_vecs[p_i],
                1.0e-5, # rtol
                1.0e-8, # atol
                constant_args # Pass the single shared vector
            )
        end_time = now()
    else:
        # Use prange
        start_time = now()
        for p_i in prange(num_runs, nogil=True, num_threads=num_threads, schedule='static'):
            cysolve_ivp_noreturn(
                results[p_i].get(), # Pass in CySolverResult pointer which is found with CySolveOutput::get()
                lotkavolterra_diffeq,
                t_start,
                t_end,
                y0_vecs[p_i],
                1.0e-5, # rtol
                1.0e-8, # atol
                constant_args # Pass the single shared vector
            )
        end_time = now()

    # Check results
    cdef CySolverResult* reference_result_ptr = results[0].get()
    # Check the reference first.
    assert reference_result_ptr.success
    assert reference_result_ptr.size > 0

    cdef CySolverResult* check_result_ptr
    cdef size_t j
    for i in range(num_runs):

        if i == 0:
            continue
        check_result_ptr = results[i].get()

        assert check_result_ptr.success
        assert check_result_ptr.size > 0
        assert reference_result_ptr.size == check_result_ptr.size

        for j in range(reference_result_ptr.size):
            assert check_result_ptr.time_domain_vec[j] == reference_result_ptr.time_domain_vec[j]
            assert check_result_ptr.solution[2 * j + 0] == reference_result_ptr.solution[2 * j + 0]
            assert check_result_ptr.solution[2 * j + 1] == reference_result_ptr.solution[2 * j + 1]


    cdef double loop_time_ms = <double>(<int>duration_cast_microseconds(end_time - start_time).count()) / 1000.0
    print(f"CyRK prange test `run_prange_constant_args_test` finished in {loop_time_ms:0.1f} ms (If you are seeing this then tests passed too).")
    print(f"\tAvg time: {loop_time_ms/num_runs:0.1f}ms/loop.")

    return loop_time_ms
