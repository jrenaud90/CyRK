from CyRK.cy.cysolver_api cimport PreEvalFunc

cdef void baseline_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void accuracy_test_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void extraoutput_test_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void lorenz_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void lorenz_extraoutput_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void lotkavolterra_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
cdef void pendulum_diffeq(double* dy_ptr, double t, double* y_ptr, char* args_ptr, PreEvalFunc pre_eval_func) noexcept nogil
