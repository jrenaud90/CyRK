from CyRK.utils.vector cimport vector
from CyRK.cy.cysolver_api cimport CySolveOutput, CySolverResult

cdef void interpolate_from_solution_list(
    double* y_result_ptr,
    int num_y,
    vector[CySolveOutput] solution_list_vec,
    int num_solutions,
    double* x_domain_ptr,
    size_t x_domain_size,
    vector[double] x_breakpoints_vec) noexcept nogil
