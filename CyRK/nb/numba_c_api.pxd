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
    
    # CyRK.cy utilities
    void cysolver_get_status_message_buffer(int status_code, char* buffer, size_t max_len)

    # Core Getters
    cpp_bool cysolver_get_success(void* ptr)
    size_t cysolver_get_size(void* ptr)
    size_t cysolver_get_num_y(void* ptr)
    size_t cysolver_get_num_dy(void* ptr)
    size_t cysolver_get_steps_taken(void* ptr)
    size_t cysolver_get_num_interpolates(void* ptr)
    int cysolver_get_status(void* ptr)
    
    # Array Pointers
    double* cysolver_get_t_ptr(void* ptr)
    double* cysolver_get_y_ptr(void* ptr)
    
    # Dense Output Methods
    int cysolver_call_call(void* ptr, double t, double* y_interp_ptr)
    int cysolver_call_call_vectorize(void* ptr, const double* t_array_ptr, size_t len_t, double* y_interp_ptr)
    
    # Diagnostic Getters
    int cysolver_get_direction(void* ptr)
    cpp_bool cysolver_get_capture_extra(void* ptr)
    cpp_bool cysolver_get_capture_dense(void* ptr)
    int cysolver_get_method(void* ptr)
    size_t cysolver_get_args_size(void* ptr)
    char* cysolver_get_args_ptr(void* ptr)
    double cysolver_get_t_now(void* ptr)
    double* cysolver_get_y_now_ptr(void* ptr)
    double* cysolver_get_dy_now_ptr(void* ptr)
    void cysolver_get_message_buffer(void* ptr, char* buffer, size_t max_len)
    
    # Destructor
    void cysolver_free(void* ptr)