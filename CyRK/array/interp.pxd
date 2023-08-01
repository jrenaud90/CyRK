# distutils: language = c++

cdef unsigned int binary_search_with_guess(double key, double[:] array, unsigned int length, unsigned int guess) nogil

cpdef (double, int) interpj(double desired_x, double[:] x_domain, double[:] dependent_values,
                            int provided_j = *) nogil

cpdef (double complex, int) interp_complexj(double desired_x, double[:] x_domain,
                                            double complex[:] dependent_values, int provided_j = *) nogil

cpdef double interp(double desired_x, double[:] x_domain, double[:] dependent_values,
                    int provided_j = *) nogil

cpdef double complex interp_complex(double desired_x, double[:] x_domain, double complex[:] dependent_values,
                                    int provided_j = *) nogil

cpdef void interp_array(double[:] desired_x_array, double[:] x_domain, double[:] dependent_values,
                        double[:] desired_dependent_array) nogil

cpdef void interp_complex_array(double[:] desired_x_array, double[:] x_domain, double complex[:] dependent_values,
                                double complex[:] desired_dependent_array) nogil