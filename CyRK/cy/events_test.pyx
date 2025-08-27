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

from CyRK.cy.common cimport CyrkErrorCodes, PreEvalFunc, MAX_STEP
from CyRK.cy.events cimport Event, EventFunc
from CyRK.cy.cysolver_api cimport cysolve_ivp_noreturn, WrapCySolverResult, ODEMethod, CySolverResult
from CyRK.cy.cysolver_test cimport lorenz_diffeq

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

    cdef int termination_int = 0
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
                event = Event(valid_func, max_allowed, termination_int)
            else:
                # Build empty event and use setup
                event = Event()
                setup_code = event.setup(valid_func, max_allowed, termination_int)
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
    if args_correct and (t >= 8.0):
        return 0.0
    else:
        return 1.0

def run_cysolver_with_events(
        bint use_dense,
        double[::1] t_eval
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

    cdef PreEvalFunc pre_eval_func = NULL
    
    # Build events
    cdef vector[Event] events_vec = vector[Event]()
    events_vec.emplace_back(lorenz_event_func_1, numeric_limits[size_t].max(), 0)
    events_vec.emplace_back(lorenz_event_func_2, numeric_limits[size_t].max(), 0)
    events_vec.emplace_back(lorenz_event_func_3, numeric_limits[size_t].max(), 0)

    cdef ODEMethod integration_method = ODEMethod.RK45
    cdef WrapCySolverResult solution = WrapCySolverResult()
    solution.build_cyresult(integration_method)
    cdef CySolverResult* solution_ptr = solution.cyresult_uptr.get()

    # Run Solver
    cysolve_ivp_noreturn(
        solution_ptr,
        lorenz_diffeq,
        t_start,
        t_end,
        y0_vec,
        rtol = 1.0e-3,
        atol = 1.0e-6,
        args_vec = args_vec,
        num_extra = 0,
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
        printf("TEST FAILED: `run_cysolver_with_events` solver did not complete successfully (use_dense = %d; t_eval_provided = %d).\n", use_dense, t_eval_provided)
        tests_passed = False
    
    # TODO Don't currently check if events executed correctly.
    
    
    return tests_passed



