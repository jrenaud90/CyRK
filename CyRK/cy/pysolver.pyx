# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from libc.string cimport memcpy
from libcpp.cmath cimport fmin, fabs
from libcpp.vector cimport vector

from CyRK.cy.cysolver_api cimport (
    find_expected_size, INF, EPS_100, CyrkErrorCodes, CyrkErrorMessages,
    ProblemConfig, RKConfig, CySolverBase)

import numpy as np
cimport numpy as cnp
cnp.import_array()

# =====================================================================================================================
# PySolver Class (holds the intergrator class and reference to the python diffeq function)
# =====================================================================================================================
cdef class PySolver(WrapCySolverResult):

    def set_pydiffeq(
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

    def set_problem_parameters(
            self,
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
        cdef ODEMethod integration_method = ODEMethod.RK45
        if method == "rk23":
            integration_method = ODEMethod.RK23
        elif method == "rk45":
            integration_method = ODEMethod.RK45
        elif method == 'dop853':
            integration_method = ODEMethod.DOP853
        else:
            raise NotImplementedError(
                "ERROR: `PySolver::set_problem_parameters` - "
                f"Unknown or unsupported integration method provided: {method}.\n"
                f"Supported methods are: RK23, RK45, DOP853."
                )

        cdef CySolverResult* cyresult_ptr = self.cyresult_uptr.get()
        if self.cyresult_uptr:
            # Already have a solution class created.
            # Is this the same integration method? If it is then we can reuse the solution class.
            if integration_method != cyresult_ptr.integrator_method:
                # Need to make a new cysolver object
                self.build_cyresult(integration_method)
        else:
            self.build_cyresult(integration_method)
        cyresult_ptr = self.cyresult_uptr.get()
        cdef ProblemConfig* base_config_ptr = cyresult_ptr.config_uptr.get()
        
        if not cyresult_ptr:
            raise RuntimeError("ERROR: `PySolver::set_problem_parameters` - CySolverResult was not constructed.")
        cdef CySolverBase* cysolver_ptr = cyresult_ptr.solver_uptr.get()
        if not cysolver_ptr:
            raise AttributeError("ERROR: `PySolver::set_problem_parameters` - CySolver not set within CySolverResult object.")

        # Parse y0
        cdef size_t i
        cdef size_t num_y  = y0.size
        cdef size_t num_dy = num_y + num_extra  
        cdef vector[double] y0_vec = vector[double](num_y)
        for i in range(num_y):
            y0_vec[i] = y0[i]
        
        # We need to set the number of ys now because we need the now state pointers. 
        # These pointers could change which memory they are pointing to if by setting the num_y later 
        # causes a realloc of the underlying vectors.
        cdef CyrkErrorCodes setup_status = cysolver_ptr.resize_num_y(num_y, num_dy)
        if setup_status != CyrkErrorCodes.NO_ERROR:
            raise Exception("ERROR: `PySolver::set_problem_parameters` - Error raised while setting number of ys: {setup_status}.")

        # Setup python diffeq
        self.set_pydiffeq(py_diffeq, args, num_y, num_dy, pass_dy_as_arg)

        # Pass python pointers to C++ classes.
        cdef DiffeqMethod diffeq_func = <DiffeqMethod>self.diffeq
        cysolver_ptr.set_cython_extension_instance(<cpy_ref.PyObject*>self, diffeq_func)

        # Pull in current state pointers from CySolver C++ object to this object so that `self.diffeq` can update the "now" attributes.
        cdef NowStatePointers solver_state = cysolver_ptr.get_now_state()
        self.set_state(&solver_state)

        # Update other configurations now
        if len(time_span) != 2:
            raise AttributeError("ERROR: `PySolver::set_problem_parameters` - Unexpected size found for the provided `time_span`.")
        cdef double t_start = time_span[0]
        cdef double t_end   = time_span[1]

        # Update configurations
        cdef RKConfig* problem_config_ptr = <RKConfig*>base_config_ptr

        # Pass python pointers to C++ classes.
        problem_config_ptr.cython_extension_class_instance = <cpy_ref.PyObject*>self
        problem_config_ptr.py_diffeq_method                = <DiffeqMethod>self.diffeq
        
        # Set required arguments.
        # diffeq_ptr - unused for PySolver.
        problem_config_ptr.t_start = t_start
        problem_config_ptr.t_end = t_end
        problem_config_ptr.y0_vec = y0_vec

        # Parse t_eval
        problem_config_ptr.t_eval_provided = False
        cdef vector[double] t_eval_vec = vector[double](0)
        if t_eval is not None:
            t_eval_vec.resize(t_eval.size)
            for i in range(t_eval.size):
                t_eval_vec[i] = t_eval[i]
            problem_config_ptr.t_eval_vec = t_eval_vec
            problem_config_ptr.t_eval_provided = True
        
        # Parse rtol
        cdef vector[double] rtols_vec = vector[double](1)
        if type(rtol) == float:
            rtols_vec[0] = rtol
        else:
            rtols_vec.resize(rtol.size)
            for y_i in range(rtol.size):
                rtols_vec[y_i] = rtol[y_i]
        problem_config_ptr.rtols = rtols_vec
        
        # Parse atol
        cdef vector[double] atols_vec = vector[double](1)
        if type(atol) == float:
            atols_vec[0] = atol
        else:
            atols_vec.resize(atol.size)
            for y_i in range(atol.size):
                atols_vec[y_i] = atol[y_i]
        problem_config_ptr.atols = atols_vec
        
        # Parse expected size
        cdef size_t expected_size_touse = expected_size
        cdef double rtol_tmp
        cdef double min_rtol = INF
        if expected_size_touse == 0:
            for i in range(rtols_vec.size()):
                rtol_tmp = rtols_vec[i]
                if rtol_tmp < EPS_100:
                    rtol_tmp = EPS_100
                min_rtol = fmin(min_rtol, rtol_tmp)
            expected_size_touse = find_expected_size(num_y, num_extra, fabs(t_end - t_start), min_rtol)
        problem_config_ptr.expected_size = expected_size_touse

        # Parse first step size
        if first_step < 0.0:
            raise AttributeError("ERROR: `PySolver::set_problem_parameters` - First step size must be a postive float (or 0.0 to use automatic finder).")
        problem_config_ptr.first_step_size = first_step
    
        # Parse maximum step size
        if max_step <= 0.0:
            raise AttributeError("ERROR: `PySolver::set_problem_parameters` - Maximum step size must be a postive float.")
        problem_config_ptr.max_step_size = max_step

        # Parse other flags
        problem_config_ptr.capture_dense_output = dense_output
        problem_config_ptr.force_retain_solver  = True # For now we are going to keep solvers it is a small memory hit but makes everything much easier to debug and avoids crashes.
        problem_config_ptr.num_extra            = num_extra
        problem_config_ptr.capture_extra        = num_extra > 0
        problem_config_ptr.max_num_steps        = max_num_steps
        problem_config_ptr.max_ram_MB           = max_ram_MB

        # Load config into cysolution
        status_code = cyresult_ptr.setup()

        if status_code != CyrkErrorCodes.NO_ERROR:
            raise Exception(
                f"ERROR: `PySolver::set_problem_parameters` - Error during config setup. Error Code: {status_code}. "
                f"Message: {CyrkErrorMessages.at(status_code).decode('utf-8')}")
    
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
        bint pass_dy_as_arg = False,
        PySolver solution_reuse = None
        ):

    # Build PySolver solution storage.
    # These cython extension classes are created as python objects so they have reference counting
    # so it is safe to return objects created in this function without worry of memory leaks.
    if solution_reuse is None:
        solution_reuse = PySolver()

    # Load in user-provided parameters
    solution_reuse.set_problem_parameters(
            py_diffeq,
            time_span,
            y0,
            method,
            t_eval,
            dense_output,
            args,
            expected_size,
            num_extra,
            first_step,
            max_step,
            rtol,
            atol,
            max_num_steps,
            max_ram_MB,
            pass_dy_as_arg)
    
    ##
    # Run the integrator!
    ##
    solution_reuse.solve()
    
    # Return the results.
    return solution_reuse
