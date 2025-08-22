# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

# From SciPy's documentation
# Each function must return a float. The solver will find an accurate value of t at which event(t, y(t)) = 0
# using a root-finding algorithm. By default, all zeros will be found. The solver looks for a sign change over each
# step, so if multiple zero crossings occur within one step, events may be missed. "

from libcpp cimport nullptr
from libcpp.limits cimport numeric_limits

from CyRK cimport CyrkErrorCodes
from CyRK.cy.events cimport Event, EventFunc

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
    for build_method in build_methods:
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
                break

    tests_passed = current_check

    return tests_passed
