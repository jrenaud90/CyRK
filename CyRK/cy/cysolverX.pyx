# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
from libc.stdio cimport printf

# =====================================================================================================================
# Import CySolverResult (container for integration results)
# =====================================================================================================================
cdef class WrapCySolverResult:

    cdef void set_cyresult_pointer(self, shared_ptr[CySolverResult] cyresult_shptr):

        # Store c++ based result and pull out key information
        self.cyresult_shptr = cyresult_shptr
        self.cyresult_ptr   = cyresult_shptr.get()
        self.size           = self.cyresult_ptr[0].size
        self.num_dy         = self.cyresult_ptr[0].num_dy

        # Convert solution to pointers and views
        self.time_ptr  = &self.cyresult_ptr[0].time_domain[0]
        self.y_ptr     = &self.cyresult_ptr[0].solution[0]
        self.time_view = <double[:self.size]>self.time_ptr
        self.y_view    = <double[:self.size * self.num_dy]>self.y_ptr

    @property
    def success(self):
        return self.cyresult_ptr.success
        
    @property
    def message(self):
        return str(self.cyresult_ptr.message_ptr, 'UTF-8')
    
    @property
    def t(self):
        return np.asarray(self.time_view, dtype=np.float64, order='C')
    
    @property
    def y(self):
        return np.asarray(self.y_view, dtype=np.float64, order='C').reshape((self.size, self.num_dy)).T
    
    @property
    def size(self):
        return self.size

# =====================================================================================================================
# PySolver Class (holds the intergrator class and reference to the python diffeq function)
# =====================================================================================================================
cdef class WrapPyDiffeq:

    def __cinit__(
            self,
            object diffeq_func,
            tuple args,
            unsigned int num_y,
            unsigned int num_dy
            ):
        
        # Install differential equation function and any additional args
        self.diffeq_func = diffeq_func
        if args is None:
            self.args = None
            self.use_args = False
        else:
            self.args = args
            self.use_args = True
        
        # Build python-safe arrays
        self.num_y  = num_y
        self.num_dy = num_dy
        self.y_now_arr   = np.empty(self.num_y, dtype=np.float64, order='C')
        self.y_now_view  = self.y_now_arr
    
    cdef void set_state(self, double* dy_ptr, double* t_ptr, double* y_ptr):
        self.dy_now_ptr = dy_ptr
        self.t_now_ptr  = t_ptr
        self.y_now_ptr  = y_ptr

    def diffeq(self):
        # Copy over pointer values to python-safe arrays
        printf("\t\tPYDIFFEQ t NOW = %e\n", self.t_now_ptr[0])
        cdef unsigned int y_i
        for y_i in range(self.num_y):
            printf("\t\tPYDIFFEQ Y NOW at %d = %e\n", y_i, self.y_now_ptr[y_i])
            self.y_now_view[y_i] = self.y_now_ptr[y_i]
        
        # Run python diffeq
        cdef double[::1] dy_view
        if self.use_args:
            dy_view = self.diffeq_func(self.t_now_ptr[0], self.y_now_arr, *self.args)
        else:
            dy_view = self.diffeq_func(self.t_now_ptr[0], self.y_now_arr)
        
        # Store results in dy pointer for the integrator
        # Note that num_dy may be larger than num_y if the user is capturing extra output during integration.
        # TODO: Could we just have the pointer point to this memory view? Perhaps if we stored the view as a self.? Or will it dealloc as soon as we leave the scope of the diffeq?
        # TODO: Also maybe a memcpy would work here too.
        printf("\t\tPYDIFFEQ LOOPING DY for %d\n", self.num_dy)
        for y_i in range(self.num_dy):
            printf("\t\tPYDIFFEQ DY PRE at %d = %e\n", y_i, self.dy_now_ptr[y_i])
            self.dy_now_ptr[y_i] = dy_view[y_i]
            printf("\t\tPYDIFFEQ DY POST at %d = %e\n", y_i, self.dy_now_ptr[y_i])

        return 1

# =====================================================================================================================
# PySolver wrapper function
# =====================================================================================================================

def pysolve_ivp(
        object py_diffeq,
        tuple time_span,
        double[::1] y0,
        str method = 'RK45',
        double[::1] t_eval = None,
        bint dense_output = False,
        tuple args = None,
        unsigned int expected_size = 0,
        unsigned int num_extra = 0,
        double first_step = 0.0,
        double max_step = INF,
        rtol = 1.0e-3,
        atol = 1.0e-6,
        size_t max_num_steps = 0,
        size_t max_ram_MB = 2000
        ):

    printf("DEBUG Point 1\n")
    # Parse method
    method = method.lower()
    cdef unsigned int integration_method = 1
    if method == "rk23":
        integration_method = 0
    elif method == "rk45":
        integration_method = 1
    elif method == 'dop853':
        integration_method = 2
    else:
        raise NotImplementedError(
            f"Unknown or unsupported integration method provided: {method}.\n"
            f"Supported methods are: RK23, RK45, DOP853."
            )
    
    printf("DEBUG Point 2\n")
    # Parse time_span
    cdef double t_start = time_span[0]
    cdef double t_end   = time_span[1]

    # Parse y0
    cdef unsigned int num_y = len(y0)
    cdef double* y0_ptr = &y0[0]
    if num_y > Y_LIMIT:
        raise AttributeError(
            f"CyRK only supports a maximum number of {Y_LIMIT} dependent variables. {num_y} were provided."
            )
    
    # Parse t_eval
    if t_eval is not None:
        raise NotImplementedError('t_eval not implemented.')
    
    # Parse dense output
    if dense_output:
        raise NotImplementedError('Dense outputs not yet supported.')
    
    # Parse num_extra
    if num_extra > (DY_LIMIT - Y_LIMIT):
        raise AttributeError(
            f"CyRK can only capture a maximum number of {DY_LIMIT - Y_LIMIT} extra outputs. {num_extra} were provided."
            )
    cdef unsigned int num_dy = num_y + num_extra
    
    printf("DEBUG Point 3\n")
    # Parse rtol
    cdef cpp_bool use_rtol_array = False
    cdef double* rtols_ptr = NULL
    cdef double[::1] rtols_view
    cdef double rtol_float = 0.0
    if type(rtol) == float:
        rtol_float = rtol
    else:
        rtols_view = np.asarray(rtol, dtype=np.float64, order='C')
        rtols_ptr = &rtols_view[0]
    
    # Parse atol
    cdef cpp_bool use_atol_array = False
    cdef double* atols_ptr = NULL
    cdef double[::1] atols_view
    cdef double atol_float = 0.0
    if type(atol) == float:
        atol_float = atol
    else:
        atols_view = np.asarray(atol, dtype=np.float64, order='C')
        atols_ptr = &atols_view[0]
    
    printf("DEBUG Point 4\n")
    # Parse expected size
    cdef double expected_size_touse = expected_size
    cdef double rtol_tmp
    cdef double min_rtol = INF
    if expected_size_touse == 0:
        if rtols_ptr:
            # rtol for each y
            for y_i in range(num_y):
                rtol_tmp = rtols_ptr[y_i]
                if rtol_tmp < EPS_100:
                    rtol_tmp = EPS_100
                min_rtol = fmin(min_rtol, rtol_tmp)
        else:
            # Only one rtol
            rtol_tmp = rtol
            if rtol_tmp < EPS_100:
                rtol_tmp = EPS_100
            min_rtol = rtol_tmp
        expected_size_touse = find_expected_size(num_y, num_extra, fabs(t_end - t_start), min_rtol)

    # Build solution storage
    printf("DEBUG Point 5\n")
    cdef shared_ptr[CySolverResult] result_ptr = make_shared[CySolverResult](num_y, num_extra, expected_size)

    # Build diffeq wrapper
    printf("DEBUG Point 6\n")
    cdef WrapPyDiffeq diffeq_wrap = WrapPyDiffeq(py_diffeq, args, num_y, num_dy)

    # Finally we can actually run the integrator!
    # The following effectively copies the functionality of cysolve_ivp. We can not directly use that function
    # because we need to tie in the python-based diffeq function (via its wrapper)
    # Build null pointers to unused arguments
    cdef DiffeqFuncType diffeq_ptr = NULL
    cdef double* args_ptr          = NULL

    printf("--> DEBUG:: diffeq ptr %p\n", <cpy_ref.PyObject*>diffeq_wrap)

    printf("DEBUG Point 7\n")
    cdef PySolver solver = PySolver(
            integration_method,
            <cpy_ref.PyObject*>diffeq_wrap,
            result_ptr,
            t_start,
            t_end,
            y0_ptr,
            num_y,
            num_extra,
            args_ptr,
            max_num_steps,
            max_ram_MB,
            rtol,
            atol,
            rtols_ptr,
            atols_ptr,
            max_step,
            first_step
        )
    printf("DEBUG Point 7b\n")

    # Get state pointers
    printf("DEBUG Point 8\n")
    cdef PySolverStatePointers state_pointers = solver.get_state_pointers()

    # Tell the diffeq wrapper class where to find the solver's state attributes
    printf("DEBUG Point 9\n")
    diffeq_wrap.set_state(
        state_pointers.dy_now_ptr,
        state_pointers.t_now_ptr,
        state_pointers.y_now_ptr
        )
    
    
    printf("POINTERS %p %p %p\n", state_pointers.dy_now_ptr, state_pointers.t_now_ptr, state_pointers.y_now_ptr)

    printf("\t\tVALUES (t) %e\n", state_pointers.t_now_ptr[0])
    printf("\t\tVALUES (dy) %e %e\n", state_pointers.dy_now_ptr[0], state_pointers.dy_now_ptr[1])
    printf("\t\tVALUES (y) %e %e\n", state_pointers.y_now_ptr[0], state_pointers.y_now_ptr[1])
    ##
    # Run the integrator!
    ##
    printf("DEBUG Point 10\n")
    solver.solve()
    
    # Wrap the solution in a python-safe wrapper class
    printf("DEBUG Point 11\n")
    cdef WrapCySolverResult pyresult = WrapCySolverResult()
    printf("DEBUG Point 12\n")
    pyresult.set_cyresult_pointer(result_ptr)
    printf("DEBUG Point 13\n")

    return pyresult
