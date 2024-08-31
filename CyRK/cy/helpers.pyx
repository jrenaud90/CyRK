# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False


cdef void interpolate_from_solution_list(
        double* y_result_ptr,
        int num_y,
        vector[CySolveOutput] solution_list_vec,
        int num_solutions,
        double* x_domain_ptr,
        size_t x_domain_size,
        vector[double] x_breakpoints_vec) noexcept nogil:
    """ Steps through multiple cysolve solutions to interpolate using a single all-domain domain. """
    
    cdef int current_sol_i = 0
    cdef double current_sol_upper_x = x_breakpoints_vec[0]
    cdef double* y_result_subptr = y_result_ptr
    cdef CySolverResult* cy_solution_ptr = solution_list_vec[0].get()

    cdef size_t x_i
    cdef double x
    
    for x_i in range(x_domain_size):
        
        x = x_domain_ptr[x_i]
        if x > current_sol_upper_x:
            # We are ready for the next solution.
            current_sol_i += 1
            current_sol_upper_x = x_breakpoints_vec[current_sol_i]
            cy_solution_ptr = solution_list_vec[current_sol_i].get()
        
        # Run cysolver interpolator
        y_result_subptr = &y_result_ptr[x_i * num_y]
        cy_solution_ptr.call(x, y_result_subptr)
