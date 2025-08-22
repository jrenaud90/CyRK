from libcpp.vector cimport vector

from CyRK.cy.cysolver_api cimport CySolverDense, CyrkErrorCodes
from CyRK.cy.events cimport EventFunc

cdef extern from "common.cpp" nogil:
    pass

cdef extern from "c_events.cpp" nogil:
    pass

cdef extern from "c_brentq.cpp" nogil:

    cdef struct OptimizeInfo:
        size_t funcalls
        size_t iterations
        CyrkErrorCodes error_num

    cdef double c_brentq(
        EventFunc func,
        double xa,
        double xb,
        double xtol,
        double rtol,
        size_t iter,
        vector[char]& func_data_vec,
        OptimizeInfo* solver_stats,
        CySolverDense* dense_func)