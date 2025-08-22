from libcpp cimport bool as cpp_bool

from CyRK cimport CyrkErrorCodes

cdef extern from "c_events.cpp" nogil:
    ctypedef double (*EventFunc)(double, double*, char*)

    size_t MAX_ALLOWED_SYS

    cdef cppclass Event:
        EventFunc check
        size_t max_allowed
        int termination_int
        CyrkErrorCodes status
        cpp_bool initialized

        Event()
        Event(
            EventFunc event_func,
            int max_allowed,
            int termination_int)

        CyrkErrorCodes setup(
            EventFunc event_func,
            int max_allowed,
            int termination_int)
