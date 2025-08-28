from libcpp.vector cimport vector

from CyRK.cy.common cimport OptimizeInfo, CyrkErrorCodes
from CyRK.cy.cysolver_api cimport CySolverDense
from CyRK.cy.events cimport EventFunc

cdef extern from "c_common.cpp" nogil:
    pass

cdef extern from "c_events.cpp" nogil:
    pass

cdef extern from "dense.cpp" nogil:
    pass

cdef extern from "c_brentq.cpp" nogil:

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