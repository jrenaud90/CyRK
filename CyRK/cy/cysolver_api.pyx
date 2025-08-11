# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from libc.math cimport floor
from libcpp.utility cimport move

import numpy as np
cimport numpy as cnp
cnp.import_array()

# =====================================================================================================================
# Import CySolverResult (container for integration results)
# =====================================================================================================================
cdef class WrapCySolverResult:

    cdef void build_cyresult(
            self,
            ODEMethod integrator_method):
        
        # Delete any CySolver Result object that may have already been set. 
        if self.cyresult_uptr:
            self.cyresult_uptr.reset()

        self.set_cyresult_pointer(make_unique[CySolverResult](integrator_method))

    cdef void set_cyresult_pointer(self, unique_ptr[CySolverResult] cyresult_uptr_):
        if not cyresult_uptr_:
            raise AttributeError("ERROR: `WrapCySolverResult::Constructor` - Provided CySolverResult is Null.")

        # Store c++ based result and pull out key information
        self.cyresult_uptr = move(cyresult_uptr_)
        cdef CySolverResult* cyresult_ptr = self.cyresult_uptr.get()

        self.finalize()
    
    cdef set_problem_config(self, ProblemConfig* new_problem_config_ptr):
        if not self.cyresult_uptr:
            raise AttributeError("ERROR: `WrapCySolverResult::set_problem_config` - CySolverResult not set.")
        self.cyresult_uptr.get().setup(new_problem_config_ptr)
    
    cpdef solve(self):
        if not self.cyresult_uptr:
            raise AttributeError("ERROR: `WrapCySolverResult::solve` - CySolverResult not set.")
        cdef CySolverResult* cyresult_ptr = self.cyresult_uptr.get()
        cyresult_ptr.solve()

        self.finalize()
    
    cpdef finalize(self):
        # Convert solution to pointers and views
        cdef CySolverResult* cyresult_ptr = self.cyresult_uptr.get()
        if cyresult_ptr:
            if cyresult_ptr.size > 0:
                self.time_ptr  = &cyresult_ptr.time_domain_vec[0]
                self.y_ptr     = &cyresult_ptr.solution[0]
                self.time_view = <double[:self.size]>self.time_ptr
                self.y_view    = <double[:self.size * self.num_dy]>self.y_ptr

    def call(self, double t):
        """ Call the dense output interpolater and return y """

        if not self.cyresult_uptr:
            raise AttributeError("ERROR: `WrapCySolverResult::call` - CySolverResult is Null.")
        cdef CySolverResult* cyresult_ptr = self.cyresult_uptr.get()
        if not cyresult_ptr.capture_dense_output:
            raise AttributeError("ERROR: `WrapCySolverResult::call` - Can not make a call to WrapCySolverResult when dense_output set to False.")

        y_interp_array = np.empty(cyresult_ptr.num_dy, dtype=np.float64, order='C')
        cdef double[::1] y_interp_view = y_interp_array
        cdef double* y_interp_ptr      = &y_interp_view[0]

        cyresult_ptr.call(t, y_interp_ptr)
        
        return y_interp_array
    
    def call_vectorize(self, double[::1] t_view):
        """ Call the dense output interpolater and return y """
        
        if not self.cyresult_uptr:
            raise AttributeError("ERROR: `WrapCySolverResult::call_vectorize` - CySolverResult is Null.")
        cdef CySolverResult* cyresult_ptr = self.cyresult_uptr.get()
        if not cyresult_ptr.capture_dense_output:
            raise AttributeError("ERROR: `WrapCySolverResult::call_vectorize` - Can not call WrapCySolverResult when dense_output set to False.")

        cdef size_t len_t = len(t_view)

        y_interp_array = np.empty(cyresult_ptr.num_dy * len_t, dtype=np.float64, order='C')
        cdef double[::1] y_interp_view = y_interp_array
        cdef double* y_interp_ptr      = &y_interp_view[0]
        cdef double* t_array_ptr       = &t_view[0]

        cyresult_ptr.call_vectorize(t_array_ptr, len_t, y_interp_ptr)
        return y_interp_array.reshape(len_t, cyresult_ptr.num_dy).T

    def print_diagnostics(self):
        
        if not self.cyresult_uptr:
            raise AttributeError("ERROR: `WrapCySolverResult::print_diagnostics` - CySolverResult is Null.")
        cdef CySolverResult* cyresult_ptr = self.cyresult_uptr.get()
    
        cdef str diagnostic_str = ''
        from CyRK import __version__
        cdef str direction_str = 'Forward'
        if cyresult_ptr.direction_flag == 0:
            direction_str = 'Backward'
        
        diagnostic_str += f'CyRK v{__version__} - WrapCySolverResult Diagnostic.\n'
        diagnostic_str += f'\n----------------------------------------------------\n'
        diagnostic_str += f'# of y:      {self.num_y}.\n'
        diagnostic_str += f'# of dy:     {self.num_dy}.\n'
        diagnostic_str += f'Success:     {self.success}.\n'
        diagnostic_str += f'Error Code:  {self.error_code}.\n'
        diagnostic_str += f'Size:        {self.size}.\n'
        diagnostic_str += f'Steps Taken: {self.steps_taken}.\n'
        diagnostic_str += f'Integrator Message:\n\t{self.message.decode("utf-8")}\n'
        diagnostic_str += f'\n----------------- CySolverResult -------------------\n'
        diagnostic_str += f'Capture Extra:         {cyresult_ptr.capture_extra}.\n'
        diagnostic_str += f'Capture Dense Output:  {cyresult_ptr.capture_dense_output}.\n'
        diagnostic_str += f'Integration Direction: {direction_str}.\n'
        diagnostic_str += f'Integration Method:    {self.integration_method}.\n'
        diagnostic_str += f'# of Interpolates:     {cyresult_ptr.num_interpolates}.\n'

        cdef CySolverBase* cysolver = cyresult_ptr.solver_uptr.get()
        cdef size_t num_y
        cdef size_t num_dy
        cdef size_t i
        cdef size_t dbl_i
        cdef size_t args_size
        cdef size_t args_size_dbls
        cdef double* args_dbl_ptr = NULL
        cdef char* args_char_ptr = NULL
        if cysolver:
            num_y  = cysolver.num_y
            num_dy = cysolver.num_dy
            diagnostic_str += f'\n------------------ CySolverBase --------------------\n'
            diagnostic_str += f'Status #:   {cyresult_ptr.status}.\n'
            diagnostic_str += f'Status:     {CyrkErrorMessages.at(cyresult_ptr.status).decode("utf-8")}.\n'
            diagnostic_str += f'# of y:     {num_y}.\n'
            diagnostic_str += f'# of dy:    {num_dy}.\n'
            diagnostic_str += f'PySolver:   {cysolver.use_pysolver}.\n'
            diagnostic_str += f'---- Current State Info ----\n'
            diagnostic_str += f't_now:      {cysolver.t_now}.\n'
            diagnostic_str += f'y_now:\n'
            for i in range(num_y):
                diagnostic_str += f'\ty{i}  = {cysolver.y_now_ptr[i]:0.5e}.\n'
            diagnostic_str += f'dy_now:\n'
            for i in range(num_dy):
                diagnostic_str += f'\tdy{i} = {cysolver.dy_now_ptr[i]:0.5e}.\n'
            diagnostic_str += f'End of Current State Info.\n'
            # We want a way to print the value of the args but we have no idea its structure or types
            # of its contents. For now just assume they are all doubles so we can display something.
            # Also display the value of the raw chars.
            diagnostic_str += f'---- Additional Argument Info ----\n'
            args_size      = cyresult_ptr.config_uptr.get().args_vec.size()
            args_char_ptr  = cyresult_ptr.config_uptr.get().args_vec.data()
            args_dbl_ptr   = <double*>args_char_ptr
            args_size_dbls = <size_t>floor(args_size / sizeof(double))
            diagnostic_str += f'args size (bytes):   {args_size}.\n'
            diagnostic_str += f'args size (doubles): {args_size_dbls}.\n'
            if not args_char_ptr:
                diagnostic_str += 'Args Pointer is Null.'
            else:
                dbl_i = 0
                if args_size > 0:
                    for i in range(args_size):
                        if i % 8 == 0:
                            # New section of 8 bytes.
                            diagnostic_str += f'\n{hex(args_char_ptr[i])}'
                        elif i % 8 == 3:
                            diagnostic_str += f' {hex(args_char_ptr[i])}\n'
                        elif i % 8 == 7:
                            diagnostic_str += f' {hex(args_char_ptr[i])}\n'
                            diagnostic_str += f'As Double: {args_dbl_ptr[dbl_i]:0.5e}\n'
                            dbl_i += 1
                        else:
                            diagnostic_str += f' {hex(args_char_ptr[i])}'
            diagnostic_str += f'End of Additional Argument Info.\n'
        else:
            diagnostic_str += f'CySolver instance was deleted or voided.\n'

        diagnostic_str += f'\n-------------- Diagnostic Complete -----------------\n'
        print(diagnostic_str)

    def __dealloc__(self):
        # Deallocate any heap allocated memory
        if self.cyresult_uptr:
            self.cyresult_uptr.reset()

    @property
    def success(self):
        if not self.cyresult_uptr:
            return None
        return self.cyresult_uptr.get().success
        
    @property
    def message(self):
        if not self.cyresult_uptr:
            return None
        # Cython automatically converts std::string to Python unicode string
        return self.cyresult_uptr.get().message.decode("utf-8")
    
    @property
    def integration_method(self):
        """Returns a string stating what integration method was used for this solution."""
        if not self.cyresult_uptr:
            return None
        return CyrkODEMethods.at(self.cyresult_uptr.get().integrator_method).decode("utf-8")
    
    @property
    def method(self):
        """Alias for `self.integration_method`."""
        return self.integration_method

    @property
    def t(self):
        return np.copy(np.asarray(self.time_view, dtype=np.float64, order='C'))
    
    @property
    def y(self):
        return np.copy(np.asarray(self.y_view, dtype=np.float64, order='C')).reshape((self.size, self.num_dy)).T
    
    @property
    def size(self):
        if not self.cyresult_uptr:
            return 0
        return self.cyresult_uptr.get().size
    
    @property
    def steps_taken(self):
        if not self.cyresult_uptr:
            return 0
        return self.cyresult_uptr.get().steps_taken
    
    @property
    def num_y(self):
        if not self.cyresult_uptr:
            return None
        return self.cyresult_uptr.get().num_y
    
    @property
    def num_dy(self):
        if not self.cyresult_uptr:
            return None
        return self.cyresult_uptr.get().num_dy
    
    @property
    def status(self):
        if not self.cyresult_uptr:
            return None
        return self.cyresult_uptr.get().status
    
    @property
    def error_code(self):
        return self.status
    
    @property
    def status_message(self):
        return CyrkErrorMessages.at(self.cyresult_uptr.get().status).decode("utf-8")
    
    def __call__(self, t):

        if not self.cyresult_uptr:
            return None

        if type(t) == cnp.ndarray:
            return self.call_vectorize(t)
        else:
            return self.call(t).reshape(self.num_dy, 1)


# =====================================================================================================================
# Create Wrapped cysolve_ivp (has various defaults)
# =====================================================================================================================
cdef void cysolve_ivp_noreturn(
            CySolverResult* solution_ptr,
            DiffeqFuncType diffeq_ptr,
            const double t_start,
            const double t_end,
            vector[double] y0_vec,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            vector[char] args_vec = vector[char](),
            size_t num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            vector[double] t_eval_vec = vector[double](),
            PreEvalFunc pre_eval_func = NULL,
            vector[double] rtols_vec = vector[double](),
            vector[double] atols_vec = vector[double](),
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept nogil:
    
    if rtols_vec.size() == 0:
        rtols_vec.resize(1)
        rtols_vec[0] = rtol
    if atols_vec.size() == 0:
        atols_vec.resize(1)
        atols_vec[0] = atol
        
    baseline_cysolve_ivp_noreturn(
        solution_ptr,
        diffeq_ptr,
        t_start,
        t_end,
        y0_vec,
        optional[size_t](expected_size),
        optional[size_t](num_extra),
        optional[vector[char]](args_vec),
        optional[size_t](max_num_steps),
        optional[size_t](max_ram_MB),
        optional[cpp_bool](dense_output),
        optional[vector[double]](t_eval_vec),
        optional[PreEvalFunc](pre_eval_func),
        optional[vector[double]](rtols_vec),
        optional[vector[double]](atols_vec),
        optional[double](max_step),
        optional[double](first_step)
        )

cdef CySolveOutput cysolve_ivp(
            DiffeqFuncType diffeq_ptr,
            const double t_start,
            const double t_end,
            vector[double] y0_vec,
            ODEMethod method = ODEMethod.RK45,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            vector[char] args_vec = vector[char](),
            size_t num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            vector[double] t_eval_vec = vector[double](),
            PreEvalFunc pre_eval_func = NULL,
            vector[double] rtols_vec = vector[double](),
            vector[double] atols_vec = vector[double](),
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept nogil:

    if rtols_vec.size() == 0:
        rtols_vec.resize(1)
        rtols_vec[0] = rtol
    if atols_vec.size() == 0:
        atols_vec.resize(1)
        atols_vec[0] = atol
    
    cdef CySolveOutput result = baseline_cysolve_ivp(
        diffeq_ptr,
        t_start,
        t_end,
        y0_vec,
        method,
        optional[size_t](expected_size),
        optional[size_t](num_extra),
        optional[vector[char]](args_vec),
        optional[size_t](max_num_steps),
        optional[size_t](max_ram_MB),
        optional[cpp_bool](dense_output),
        optional[vector[double]](t_eval_vec),
        optional[PreEvalFunc](pre_eval_func),
        optional[vector[double]](rtols_vec),
        optional[vector[double]](atols_vec),
        optional[double](max_step),
        optional[double](first_step)
        )

    return move(result)

cdef CySolveOutput cysolve_ivp_gil(
            DiffeqFuncType diffeq_ptr,
            const double t_start,
            const double t_end,
            vector[double] y0_vec,
            ODEMethod method = ODEMethod.RK45,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            vector[char] args_vec = vector[char](),
            size_t num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            vector[double] t_eval_vec = vector[double](),
            PreEvalFunc pre_eval_func = NULL,
            vector[double] rtols_vec = vector[double](),
            vector[double] atols_vec = vector[double](),
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept:
    
    if rtols_vec.size() == 0:
        rtols_vec.resize(1)
        rtols_vec[0] = rtol
    if atols_vec.size() == 0:
        atols_vec.resize(1)
        atols_vec[0] = atol

    cdef CySolveOutput result = baseline_cysolve_ivp(
        diffeq_ptr,
        t_start,
        t_end,
        y0_vec,
        method,
        optional[size_t](expected_size),
        optional[size_t](num_extra),
        optional[vector[char]](args_vec),
        optional[size_t](max_num_steps),
        optional[size_t](max_ram_MB),
        optional[cpp_bool](dense_output),
        optional[vector[double]](t_eval_vec),
        optional[PreEvalFunc](pre_eval_func),
        optional[vector[double]](rtols_vec),
        optional[vector[double]](atols_vec),
        optional[double](max_step),
        optional[double](first_step)
        )

    return move(result)
