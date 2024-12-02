# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from libc.math cimport floor

import numpy as np
cimport numpy as cnp
cnp.import_array()

# =====================================================================================================================
# Import CySolverResult (container for integration results)
# =====================================================================================================================
cdef class WrapCySolverResult:

    cdef void set_cyresult_pointer(self, shared_ptr[CySolverResult] cyresult_shptr):

        # Store c++ based result and pull out key information
        self.cyresult_shptr = cyresult_shptr
        self.cyresult_ptr   = cyresult_shptr.get()

        # Convert solution to pointers and views
        if self.cyresult_ptr.size > 0:
            self.time_ptr  = &self.cyresult_ptr.time_domain_vec[0]
            self.y_ptr     = &self.cyresult_ptr.solution[0]
            self.time_view = <double[:self.size]>self.time_ptr
            self.y_view    = <double[:self.size * self.num_dy]>self.y_ptr
    
    def call(self, double t):
        """ Call the dense output interpolater and return y """

        if not self.cyresult_ptr.capture_dense_output:
            raise AttributeError("Can not call WrapCySolverResult when dense_output set to False.")

        y_interp_array = np.empty(self.cyresult_ptr.num_dy, dtype=np.float64, order='C')
        cdef double[::1] y_interp_view = y_interp_array
        cdef double* y_interp_ptr      = &y_interp_view[0]

        self.cyresult_ptr.call(t, y_interp_ptr)
        return y_interp_array
    
    def call_vectorize(self, double[::1] t_view):
        """ Call the dense output interpolater and return y """

        if not self.cyresult_ptr.capture_dense_output:
            raise AttributeError("Can not call WrapCySolverResult when dense_output set to False.")

        cdef size_t len_t = len(t_view)

        y_interp_array = np.empty(self.cyresult_ptr.num_dy * len_t, dtype=np.float64, order='C')
        cdef double[::1] y_interp_view = y_interp_array
        cdef double* y_interp_ptr = &y_interp_view[0]
        cdef double* t_array_ptr  = &t_view[0]

        self.cyresult_ptr.call_vectorize(t_array_ptr, len_t, y_interp_ptr)
        return y_interp_array.reshape(len_t, self.cyresult_ptr.num_dy).T

    def print_diagnostics(self):

        cdef str diagnostic_str = ''
        from CyRK import __version__
        cdef str direction_str = 'Forward'
        if self.cyresult_ptr.direction_flag == 0:
            direction_str = 'Backward'
        
        diagnostic_str += f'CyRK v{__version__} - WrapCySolverResult Diagnostic.\n'
        diagnostic_str += f'\n----------------------------------------------------\n'
        diagnostic_str += f'# of y:      {self.num_y}.\n'
        diagnostic_str += f'# of dy:     {self.num_dy}.\n'
        diagnostic_str += f'Success:     {self.success}.\n'
        diagnostic_str += f'Error Code:  {self.error_code}.\n'
        diagnostic_str += f'Size:        {self.size}.\n'
        diagnostic_str += f'Steps Taken: {self.steps_taken}.\n'
        diagnostic_str += f'Message:\n\t{self.message}\n'
        diagnostic_str += f'\n----------------- CySolverResult -------------------\n'
        diagnostic_str += f'Capture Extra:         {self.cyresult_ptr.capture_extra}.\n'
        diagnostic_str += f'Capture Dense Output:  {self.cyresult_ptr.capture_dense_output}.\n'
        diagnostic_str += f'Integration Direction: {direction_str}.\n'
        diagnostic_str += f'Integration Method:    {self.cyresult_ptr.integrator_method}.\n'
        diagnostic_str += f'# of Interpolates:     {self.cyresult_ptr.num_interpolates}.\n'

        cdef CySolverBase* cysolver = self.cyresult_ptr.solver_uptr.get()
        cdef size_t num_y
        cdef size_t num_dy
        cdef size_t i
        cdef size_t args_size
        cdef size_t args_size_dbls
        cdef double* args_dbl_ptr
        if cysolver:
            num_y  = cysolver.num_y
            num_dy = cysolver.num_dy
            diagnostic_str += f'\n------------------ CySolverBase --------------------\n'
            diagnostic_str += f'Status:     {cysolver.status}.\n'
            diagnostic_str += f'# of y:     {num_y}.\n'
            diagnostic_str += f'# of dy:    {num_dy}.\n'
            diagnostic_str += f'PySolver:   {cysolver.use_pysolver}.\n'
            diagnostic_str += f't_now:      {cysolver.t_now}.\n'
            diagnostic_str += f'y_now:\n'
            for i in range(num_y):
                diagnostic_str += f'\ty{i}  = {cysolver.y_now[i]:0.5e}.\n'
            diagnostic_str += f'dy_now:\n'
            for i in range(num_dy):
                diagnostic_str += f'\tdy{i} = {cysolver.dy_now[i]:0.5e}.\n'
            args_size      = cysolver.size_of_args
            args_size_dbls = <size_t>floor(args_size / sizeof(double))
            args_dbl_ptr   = <double*>cysolver.args_ptr
            diagnostic_str += f'args size (bytes):   {args_size}.\n'
            diagnostic_str += f'args size (doubles): {args_size_dbls}.\n'
            if args_size_dbls > 0:
                diagnostic_str += f'args (as doubles):\n'
                for i in range(args_size_dbls):
                    diagnostic_str += f'\targ{i} = {args_dbl_ptr[i]:0.5e}.\n'
        else:
            diagnostic_str += f'CySolverBase instance was deleted or voided.\n'

        diagnostic_str += f'\n-------------- Diagnostic Complete -----------------\n'
        print(diagnostic_str)

    @property
    def success(self):
        return self.cyresult_ptr.success
        
    @property
    def message(self):
        return str(self.cyresult_ptr.message_ptr, 'UTF-8')
    
    @property
    def t(self):
        return np.copy(np.asarray(self.time_view, dtype=np.float64, order='C'))
    
    @property
    def y(self):
        return np.copy(np.asarray(self.y_view, dtype=np.float64, order='C')).reshape((self.size, self.num_dy)).T
    
    @property
    def size(self):
        return self.cyresult_ptr.size
    
    @property
    def steps_taken(self):
        return self.cyresult_ptr.steps_taken
    
    @property
    def num_y(self):
        return self.cyresult_ptr.num_y
    
    @property
    def num_dy(self):
        return self.cyresult_ptr.num_dy
    
    @property
    def error_code(self):
        return self.cyresult_ptr.error_code
    
    def __call__(self, t):

        if type(t) == cnp.ndarray:
            return self.call_vectorize(t)
        else:
            return self.call(t).reshape(self.cyresult_ptr.num_dy, 1)


# =====================================================================================================================
# Create Wrapped cysolve_ivp (has various defaults)
# =====================================================================================================================
cdef void cysolve_ivp_noreturn(
            shared_ptr[CySolverResult] solution_sptr,
            DiffeqFuncType diffeq_ptr,
            const double* t_span_ptr,
            const double* y0_ptr,
            const size_t num_y,
            int method = 1,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            char* args_ptr = NULL,
            size_t size_of_args = 0,
            size_t num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            double* t_eval = NULL,
            size_t len_t_eval = 0,
            PreEvalFunc pre_eval_func = NULL,
            double* rtols_ptr = NULL,
            double* atols_ptr = NULL,
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept nogil:
    baseline_cysolve_ivp_noreturn(
        solution_sptr,
        diffeq_ptr,
        t_span_ptr,
        y0_ptr,
        num_y,
        method,
        expected_size,
        num_extra,
        args_ptr,
        size_of_args,
        max_num_steps,
        max_ram_MB,
        dense_output,
        t_eval,
        len_t_eval,
        pre_eval_func,
        rtol,
        atol,
        rtols_ptr,
        atols_ptr,
        max_step,
        first_step
        )

cdef CySolveOutput cysolve_ivp(
            DiffeqFuncType diffeq_ptr,
            const double* t_span_ptr,
            const double* y0_ptr,
            const size_t num_y,
            int method = 1,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            char* args_ptr = NULL,
            size_t size_of_args = 0,
            size_t num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            double* t_eval = NULL,
            size_t len_t_eval = 0,
            PreEvalFunc pre_eval_func = NULL,
            double* rtols_ptr = NULL,
            double* atols_ptr = NULL,
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept nogil:
    cdef CySolveOutput result = baseline_cysolve_ivp(
        diffeq_ptr,
        t_span_ptr,
        y0_ptr,
        num_y,
        method,
        expected_size,
        num_extra,
        args_ptr,
        size_of_args,
        max_num_steps,
        max_ram_MB,
        dense_output,
        t_eval,
        len_t_eval,
        pre_eval_func,
        rtol,
        atol,
        rtols_ptr,
        atols_ptr,
        max_step,
        first_step
        )

    return result

cdef CySolveOutput cysolve_ivp_gil(
            DiffeqFuncType diffeq_ptr,
            const double* t_span_ptr,
            const double* y0_ptr,
            const size_t num_y,
            int method = 1,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            char* args_ptr = NULL,
            size_t size_of_args = 0,
            size_t num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            double* t_eval = NULL,
            size_t len_t_eval = 0,
            PreEvalFunc pre_eval_func = NULL,
            double* rtols_ptr = NULL,
            double* atols_ptr = NULL,
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept:
    
    cdef CySolveOutput result = baseline_cysolve_ivp(
        diffeq_ptr,
        t_span_ptr,
        y0_ptr,
        num_y,
        method,
        expected_size,
        num_extra,
        args_ptr,
        size_of_args,
        max_num_steps,
        max_ram_MB,
        dense_output,
        t_eval,
        len_t_eval,
        pre_eval_func,
        rtol,
        atol,
        rtols_ptr,
        atols_ptr,
        max_step,
        first_step
        )

    return result
