# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

# From SciPy's documentation
# Each function must return a float. The solver will find an accurate value of t at which event(t, y(t)) = 0
# using a root-finding algorithm. By default, all zeros will be found. The solver looks for a sign change over each
# step, so if multiple zero crossings occur within one step, events may be missed. "

from libc.stdio cimport printf

from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libcpp.limits cimport numeric_limits

from CyRK.cy.common cimport CyrkErrorCodes, PreEvalFunc, MAX_STEP, DiffeqFuncType
from CyRK.cy.events cimport Event, EventFunc
from CyRK.cy.cysolver_api cimport cysolve_ivp_noreturn, WrapCySolverResult, ODEMethod, CySolverResult
from CyRK.cy.cysolver_test cimport lorenz_diffeq, lorenz_extraoutput_diffeq

cdef double valid_func_1(double t, double* y_ptr, char* unused_args) noexcept nogil:

    # Check if y0 is negative
    if y_ptr[0] < 0.0:
        return 0.0
    else:
        return 1.0

cdef double valid_func_2(double t, double* y_ptr, char* unused_args) noexcept nogil:

    # Check if y0 is positive
    if y_ptr[0] > 0.0:
        return 0.0
    else:
        return 1.0

cdef double valid_func_3(double t, double* y_ptr, char* unused_args) noexcept nogil:

    # Check if y1 is negative
    if y_ptr[1] < 0.0:
        return 0.0
    else:
        return 1.0


def build_event_wrapper_test():

    cdef bint tests_passed = False

    cdef EventFunc[3] event_func_array = [valid_func_1, valid_func_2, valid_func_3]
    cdef EventFunc valid_func = NULL

    cdef Event event

    cdef int build_method = -1
    cdef int[2] build_methods = [0, 1]

    cdef int direction = 0
    cdef size_t max_allowed = numeric_limits[size_t].max()

    cdef CyrkErrorCodes setup_code = CyrkErrorCodes.NO_ERROR
    cdef bint current_check = True
    cdef int build_method_tracker = 0
    cdef int func_tracker = 0
    for build_method in build_methods:
        func_tracker = 0
        for valid_func in event_func_array:                
            if build_method == 0:
                # Use constructor with all args
                event = Event(valid_func, max_allowed, direction)
            else:
                # Build empty event and use setup
                event = Event()
                setup_code = event.setup(valid_func, max_allowed, direction)
                current_check = current_check and (setup_code == CyrkErrorCodes.NO_ERROR)
            current_check = current_check and event.initialized and (event.status == CyrkErrorCodes.NO_ERROR)
            if not current_check:
                printf("TEST FAILED: `build_event_wrapper_test` - building events failed at build method %d and event func %d.", build_method_tracker, func_tracker)
                break
            func_tracker += 1
        build_method_tracker += 1

    tests_passed = current_check

    return tests_passed


cdef double lorenz_event_func_1(double t, double* y_ptr, char* unused_args) noexcept nogil:

    # Check if t greater than or equal to 5.0
    if t >= 5.0:
        return 0.0
    else:
        return 1.0

cdef double lorenz_event_func_2(double t, double* y_ptr, char* unused_args) noexcept nogil:

    # Check y values.
    # In the time span [0,10]: 
    #    y_0  starts at 1, spikes then goes below zero and oscillates with a min below -10. Have this return if y_0 < -10
    if y_ptr[0] < -10.0:
        return 0.0
    elif y_ptr[2] > 30.0:
        return 0.0
    else:
        return 1.0

cdef double lorenz_event_func_3(double t, double* y_ptr, char* args) noexcept nogil:

    # Use arguments to set threshold
    cdef double* args_dbl_ptr = <double*>args

    # We won't actually use the args but lets just check they are correct.
    cdef cpp_bool args_correct = False
    if args_dbl_ptr[0] == 10.0 and args_dbl_ptr[1] == 28.0 and args_dbl_ptr[2] == 8.0 / 3.0:
        args_correct = True

    # Then return events if args are correct and t greater than 8
    if args_correct:
        return 0.0
    else:
        return 1.0


def run_cysolver_with_events(
        bint use_dense,
        double[::1] t_eval,
        bint use_termination,
        bint capture_extra
    ):

    cdef bint tests_passed = True

    # Inputs for lorenz diffeq
    cdef size_t num_y = 3
    cdef vector[double] y0_vec = vector[double](num_y)
    y0_vec[0] = 1.0
    y0_vec[1] = 0.0
    y0_vec[2] = 0.0

    cdef vector[double] t_eval_vec = vector[double]()
    cdef cpp_bool t_eval_provided = False
    cdef size_t i
    if t_eval.size > 0:
        t_eval_provided = True
        for i in range(t_eval.size):
            t_eval_vec.push_back(t_eval[i])

    cdef double t_start = 0.0
    cdef double t_end = 10.0

    cdef vector[char] args_vec = vector[char](3 * sizeof(double))
    args_dbl_ptr = <double*>args_vec.data()
    args_dbl_ptr[0] = 10.0
    args_dbl_ptr[1] = 28.0
    args_dbl_ptr[2] = 8.0 / 3.0

    cdef size_t num_dy = 3

    cdef PreEvalFunc pre_eval_func = NULL
    
    # Build events
    cdef vector[Event] events_vec = vector[Event]()
    cdef size_t max_allowed = numeric_limits[size_t].max()
    cdef int direction = 0
    if use_termination:
        events_vec.emplace_back(lorenz_event_func_1, 1, 0)
    else:
        events_vec.emplace_back(lorenz_event_func_1, max_allowed, direction)
    events_vec.emplace_back(lorenz_event_func_2, max_allowed, direction)
    events_vec.emplace_back(lorenz_event_func_3, max_allowed, direction)

    cdef ODEMethod integration_method = ODEMethod.RK45
    cdef WrapCySolverResult solution = WrapCySolverResult()
    solution.build_cyresult(integration_method)
    cdef CySolverResult* solution_ptr = solution.cyresult_uptr.get()

    cdef DiffeqFuncType diffeq = lorenz_diffeq
    cdef size_t num_extra = 0
    if capture_extra:
        diffeq = lorenz_extraoutput_diffeq
        num_extra = 3
        num_dy = 6

    # Run Solver
    cysolve_ivp_noreturn(
        solution_ptr,
        diffeq,
        t_start,
        t_end,
        y0_vec,
        rtol = 1.0e-3,
        atol = 1.0e-6,
        args_vec = args_vec,
        num_extra = num_extra,
        max_num_steps = 100000,
        max_ram_MB = 2000,
        dense_output = use_dense,
        t_eval_vec = t_eval_vec,
        pre_eval_func = pre_eval_func,
        events_vec = events_vec,
        rtols_vec = vector[double](),
        atols_vec = vector[double](),
        max_step = MAX_STEP,
        first_step = 0.0,
        expected_size = 128
        )
    
    solution.finalize()
    if not solution_ptr.success:
        printf("TEST FAILED: `run_cysolver_with_events` solver did not complete successfully.\n")
        tests_passed = False
    
    # Check that solution storage is properly setup to handle events
    if solution_ptr.num_events != 3:
        printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.num_events != 3 (%d).\n", solution_ptr.num_events)
        tests_passed = False
    if solution_ptr.event_times.size() != 3:
        printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.event_times.size() != 3 (%d).\n", solution_ptr.event_times.size())
        tests_passed = False
    if solution_ptr.event_states.size() != 3:
        printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.event_states.size() != 3 (%d).\n", solution_ptr.event_states.size())
        tests_passed = False

    # Check Solution event data
    if solution_ptr.num_events != 3:
        printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.num_events != 3 (%d).\n", solution_ptr.num_events)
        tests_passed = False
    
    # Check CySolver for correct event data
    if solution_ptr.solver_uptr.get().num_events != 3:
        printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.solver_uptr.get().num_events != 3 (%d).\n", solution_ptr.solver_uptr.get().num_events)
        tests_passed = False
    
    # Check results of event run.
    if solution_ptr.event_times.size() != 3:
        printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.event_times.size() != 3 (%d).\n", solution_ptr.event_times.size())
        tests_passed = False
    if solution_ptr.event_states.size() != 3:
        printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.event_states.size() != 3 (%d).\n", solution_ptr.event_states.size())
        tests_passed = False
    
    cdef size_t event_i
    for event_i in range(3):
        printf("INFO: `run_cysolver_with_events` - solution_ptr.event_times[%d].size() = %d\n", event_i, solution_ptr.event_times[event_i].size())
        if solution_ptr.event_times[event_i].size() == 0:
            printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.event_times[%d].size() == 0.\n", event_i)
            tests_passed = False
            break
        
        printf("INFO: `run_cysolver_with_events` - solution_ptr.event_states[%d].size() = %d\n", event_i, solution_ptr.event_states[event_i].size())
        if solution_ptr.event_states[event_i].size() == 0:
            printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.event_states[%d].size() == 0.\n", event_i)
            tests_passed = False
            break
    
        if solution_ptr.event_states[event_i].size() != num_dy * solution_ptr.event_times[event_i].size():
            printf("TEST FAILED: `run_cysolver_with_events` - solution_ptr.event_states[%d].size() / num_dy (%d) != event_times.\n", event_i, num_dy)
            tests_passed = False
            break

    cdef double time_check = 0.0
    if use_termination:
        if not solution_ptr.event_terminated:
            printf("TEST FAILED: `run_cysolver_with_events` - solution.event_terminated is False.\n")
            tests_passed = False

        time_check = solution_ptr.time_domain_vec.back()

        # Surprisingly scipy's root finding method really does not get very close to 5.0 so we will check if its above 5.1; for this example it should be ~5.035
        if time_check > 5.1:
            printf("TEST FAILED: `run_cysolver_with_events` - Solver did not terminated when requested: t_end = %f.\n", time_check)
            tests_passed = False
    
    if not tests_passed:
        printf("`run_cysolver_with_events` - Tests Failed. use_dense = %d; t_eval_provided = %d; use_termination = %d\n", use_dense, t_eval_provided, use_termination)
    
    return solution, tests_passed



