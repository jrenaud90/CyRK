from libcpp cimport bool as cpp_bool

from CyRK.cy.cysolver_api cimport DiffeqFuncType, PreEvalFunc, Event

# Numba API
cdef extern from "numba_c_api_.cpp" nogil:
    void* numba_cysolve_ivp(
        DiffeqFuncType diffeq_ptr,
        double t_start,
        double t_end,
        double* y0_ptr, size_t y0_len,
        int integration_method_int,
        size_t expected_size,
        size_t num_extra,
        char* args_ptr, size_t args_len,
        size_t max_num_steps,
        size_t max_ram_MB,
        cpp_bool capture_dense_output,
        double* t_eval_ptr, size_t t_eval_len,
        PreEvalFunc pre_eval_func,
        Event* events_ptr, size_t events_len,
        double* rtols_ptr, size_t rtols_len,
        double* atols_ptr, size_t atols_len,
        double max_step_size,
        double first_step_size,
        cpp_bool force_retain_solver)
    
    
    cpp_bool cysolver_get_success(void* ptr)
    size_t cysolver_get_size(void* ptr)
    size_t cysolver_get_num_dy(void* ptr)
    double* cysolver_get_t_ptr(void* ptr)
    double* cysolver_get_y_ptr(void* ptr)
    int cysolver_get_status(void* ptr)
    void cysolver_free(void* ptr)

