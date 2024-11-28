# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.string cimport memcpy
from libcpp.cmath cimport fmin, fabs

from CyRK.utils.memory cimport make_shared
from CyRK.cy.cysolver_api cimport find_expected_size, WrapCySolverResult, INF, EPS_100

import numpy as np
cimport numpy as cnp
cnp.import_array()

# =====================================================================================================================
# PySolver Class (holds the intergrator class and reference to the python diffeq function)
# =====================================================================================================================
cdef class WrapPyDiffeq:

    def __cinit__(
            self,
            object diffeq_func,
            tuple args,
            size_t num_y,
            size_t num_dy,
            bint pass_dy_as_arg = False
            ):
        
        # Install differential equation function and any additional args
        self.diffeq_func = diffeq_func
        if args is None:
            self.args     = None
            self.use_args = False
        else:
            if len(args) == 0:
                # Empty tuple provided. Don't use args.
                self.args     = None
                self.use_args = False
            else:
                self.args     = args
                self.use_args = True
        
        # Build python-safe arrays
        self.num_y  = num_y
        self.num_dy = num_dy

        if pass_dy_as_arg:
            self.pass_dy_as_arg = True
        else:
            self.pass_dy_as_arg = False
    
    cdef void set_state(self, NowStatePointers* solver_state_ptr) noexcept:
        
        self.t_now_ptr  = solver_state_ptr.t_now_ptr
        self.y_now_ptr  = solver_state_ptr.y_now_ptr
        self.dy_now_ptr = solver_state_ptr.dy_now_ptr

        # Create memoryviews of the pointers
        self.y_now_view  = <double[:self.num_y]>self.y_now_ptr

        # Create numpy arrays which will be passed to the python diffeq.
        # We need to make sure that this is a not a new ndarray, but one that points to the same data. 
        # That is why we use `PyArray_SimpleNewFromData` instead of a more simple `asarray`.
        # Note that it is not safe to return these arrays outside of this class because they may get deallocated while
        # the numpy array still points to the underlying memory.
        cdef cnp.npy_intp[1] shape
        cdef cnp.npy_intp* shape_ptr = &shape[0]
        shape_ptr[0] = <cnp.npy_intp>self.num_y
        
        self.y_now_arr = cnp.PyArray_SimpleNewFromData(1, shape_ptr, cnp.NPY_DOUBLE, self.y_now_ptr)
        
        # Do the same for dy if the user provided the appropriate kind of differential equation.
        if self.pass_dy_as_arg:
            self.dy_now_view = <double[:self.num_dy]>self.dy_now_ptr
            shape[0]         = <cnp.npy_intp>self.num_dy  # dy may have a larger shape than y
            self.dy_now_arr  = cnp.PyArray_SimpleNewFromData(1, shape_ptr, cnp.NPY_DOUBLE, self.dy_now_ptr)   

    cdef void diffeq(self) noexcept:
        # Run python diffeq
        if self.pass_dy_as_arg:
            if self.use_args:
                self.diffeq_func(self.dy_now_arr, self.t_now_ptr[0], self.y_now_arr, *self.args)
            else:
                self.diffeq_func(self.dy_now_arr, self.t_now_ptr[0], self.y_now_arr)
        else:
            if self.use_args:
                self.dy_now_view = self.diffeq_func(self.t_now_ptr[0], self.y_now_arr, *self.args)
            else:
                self.dy_now_view = self.diffeq_func(self.t_now_ptr[0], self.y_now_arr)
            # Since we do not have a static dy that we can pass to the function and use in the solver we must copy over
            # the values from the newly created dy memory view
            # Note that num_dy may be larger than num_y if the user is capturing extra output during integration.
            memcpy(self.dy_now_ptr, &self.dy_now_view[0], sizeof(double) * self.num_dy)

# =====================================================================================================================
# PySolver wrapper function
# =====================================================================================================================

def pysolve_ivp(
        object py_diffeq,
        tuple time_span,
        const double[::1] y0,
        str method = 'RK45',
        const double[::1] t_eval = None,
        bint dense_output = False,
        tuple args = None,
        size_t expected_size = 0,
        size_t num_extra = 0,
        double first_step = 0.0,
        double max_step = INF,
        rtol = 1.0e-3,
        atol = 1.0e-6,
        size_t max_num_steps = 0,
        size_t max_ram_MB = 2000,
        bint pass_dy_as_arg = False
        ):

    # Parse method
    method = method.lower()
    cdef int integration_method = 1
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
    
    # Parse time_span
    cdef double t_start = time_span[0]
    cdef double t_end   = time_span[1]
    cdef cpp_bool direction_flag = (t_end - t_start) >= 0

    # Parse y0
    cdef size_t num_y   = len(y0)
    cdef const double* y0_ptr = &y0[0]
    
    # Parse t_eval
    cdef const double* t_eval_ptr = NULL
    cdef cpp_bool t_eval_provided = False
    cdef size_t len_t_eval = 0
    if t_eval is not None:
        t_eval_provided = True
        len_t_eval = len(t_eval)
        t_eval_ptr = &t_eval[0]
    
    # Parse num_extra
    cdef size_t num_dy = num_y + num_extra
    
    # Parse rtol
    cdef double* rtols_ptr = NULL
    cdef double[::1] rtols_view
    cdef double rtol_float = 0.0
    if type(rtol) == float:
        rtol_float = rtol
        rtol_float = fabs(rtol_float)
    else:
        rtols_view = np.abs(np.asarray(rtol, dtype=np.float64, order='C'))
        rtols_ptr = &rtols_view[0]
    
    # Parse atol
    cdef double* atols_ptr = NULL
    cdef double[::1] atols_view
    cdef double atol_float = 0.0
    if type(atol) == float:
        atol_float = atol
        atol_float = fabs(atol_float)
    else:
        atols_view = np.abs(np.asarray(atol, dtype=np.float64, order='C'))
        atols_ptr = &atols_view[0]
    
    # Parse expected size
    cdef size_t expected_size_touse = expected_size
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
    
    # Parse first step size
    if first_step < 0.0:
        raise AttributeError("First step size must be a postive float (or 0.0 to use automatic finder).")

    # Parse maximum step size
    if max_step <= 0.0:
        raise AttributeError("Maximum step size must be a postive float.")

    # Build solution storage
    cdef shared_ptr[CySolverResult] solution_sptr = make_shared[CySolverResult](
        num_y,
        num_extra,
        expected_size,
        t_end,
        direction_flag,
        dense_output,
        t_eval_provided)

    # Build diffeq wrapper
    cdef WrapPyDiffeq diffeq_wrap = WrapPyDiffeq(py_diffeq, args, num_y, num_dy, pass_dy_as_arg=pass_dy_as_arg)
    cdef DiffeqMethod diffeq_func = <DiffeqMethod>diffeq_wrap.diffeq

    # Finally we can actually run the integrator!
    # The following effectively copies the functionality of cysolve_ivp. We can not directly use that function
    # because we need to tie in the python-based diffeq function (via its wrapper)

    # We need to heap allocate the PySolver class instance otherwise it can get garbage collected while the solver
    # is running.
    cdef PySolver* solver = new PySolver(
            integration_method,
            <cpy_ref.PyObject*>diffeq_wrap,
            diffeq_func,
            solution_sptr,
            t_start,
            t_end,
            y0_ptr,
            num_y,
            expected_size,
            num_extra,
            max_num_steps,
            max_ram_MB,
            dense_output,
            t_eval_ptr,
            len_t_eval,
            rtol_float,
            atol_float,
            rtols_ptr,
            atols_ptr,
            max_step,
            first_step
        )

    # Get pointers to the solver so that the Python differential equation can use and update its state variables (t_now, y_now, dy_now).
    cdef NowStatePointers solver_state = solution_sptr.get().solver_uptr.get().get_now_state()
    diffeq_wrap.set_state(&solver_state)
    
    ##
    # Run the integrator!
    ##
    solver.solve()
    
    # Wrap the solution in a python-safe wrapper class
    cdef WrapCySolverResult pyresult = WrapCySolverResult()
    pyresult.set_cyresult_pointer(solution_sptr)

    del solver

    return pyresult
