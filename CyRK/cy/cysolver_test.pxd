from libcpp cimport bool as cpp_bool

from libc.math cimport sin, cos, fabs, fmin, fmax

from CyRK.cy.cysolver_api cimport cysolve_ivp, WrapCySolverResult, DiffeqFuncType,MAX_STEP, CySolveOutput, PreEvalFunc

cdef void baseline_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void accuracy_test_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void extraoutput_test_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void lorenz_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void lorenz_extraoutput_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void lotkavolterra_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, const void* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
