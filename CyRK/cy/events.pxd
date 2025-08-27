from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector

from CyRK.cy.common cimport CyrkErrorCodes

cdef extern from "c_events.cpp" nogil:
    ctypedef double (*EventFunc)(double, double*, char*)

    size_t MAX_ALLOWED_SYS

    cdef cppclass Event:
        EventFunc check
        size_t max_allowed
        size_t current_count
        CyrkErrorCodes status
        double last_root
        int direction
        cpp_bool is_active
        cpp_bool initialized
        vector[double] y_at_root_vec

        Event()
        Event(
            EventFunc event_func,
            size_t max_allowed,
            int direction)

        CyrkErrorCodes setup(
            EventFunc event_func,
            size_t max_allowed,
            int direction)
