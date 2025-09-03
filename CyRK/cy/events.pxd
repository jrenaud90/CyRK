from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector

cimport cpython.ref as cpy_ref

from CyRK.cy.common cimport CyrkErrorCodes, MAX_SIZET_SIZE
from CyRK.cy.pysolver_cyhook cimport PyEventMethod

cdef extern from "c_brentq.cpp" nogil:
    pass

cdef extern from "c_events.cpp" nogil:
    ctypedef double (*EventFunc)(double, double*, char*)

    cdef cppclass Event:
        EventFunc check
        size_t max_allowed
        size_t current_count
        size_t pyevent_index
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
        double py_check(
            double t,
            double* y_ptr,
            char* arg_ptr)
        CyrkErrorCodes set_cython_extension_instance(
            cpy_ref.PyObject* cython_extension_class_instance,
            PyEventMethod pyevent_method
            )

    ctypedef double (*EventFuncWithInst)(Event*, double, double*, char*)