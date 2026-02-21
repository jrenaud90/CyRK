// numba_c_api.cpp (or append to an existing .cpp file)
#include "cysolution.hpp"
#include "cysolve.hpp"

// The Execution Wrapper
void* numba_cysolve_ivp(
        DiffeqFuncType diffeq_ptr,
        double t_start,
        double t_end,
        double* y0_ptr, size_t y0_len,
        ODEMethod integration_method,
        size_t expected_size,
        size_t num_extra,
        char* args_ptr, size_t args_len,
        size_t max_num_steps,
        size_t max_ram_MB,
        bool capture_dense_output,
        double* t_eval_ptr, size_t t_eval_len,
        PreEvalFunc pre_eval_func,
        Event* events_ptr, size_t events_len,
        double* rtols_ptr, size_t rtols_len,
        double* atols_ptr, size_t atols_len,
        double max_step_size,
        double first_step_size,
        bool force_retain_solver
    )
{
    // Reconstruct std::vectors from the raw pointers provided by Numba
    std::vector<double> y0_vec(y0_ptr, y0_ptr + y0_len);
    std::vector<char> args_vec(args_ptr, args_ptr + args_len);
    std::vector<double> t_eval_vec(t_eval_ptr, t_eval_ptr + t_eval_len);
    std::vector<Event> events_vec(events_ptr, events_ptr + events_len);
    std::vector<double> rtols(rtols_ptr, rtols_ptr + rtols_len);
    std::vector<double> atols(atols_ptr, atols_ptr + atols_len);

    // For args, we currently only support users passing in doubles. But cysolve_ivp expects a char

    // Call the baseline solver
    std::unique_ptr<CySolverResult> result = baseline_cysolve_ivp(
        diffeq_ptr,
        t_start,
        t_end,
        y0_vec,
        integration_method,
        expected_size,
        num_extra,
        args_vec,
        max_num_steps,
        max_ram_MB,
        capture_dense_output,
        t_eval_vec,
        pre_eval_func,
        events_vec,
        rtols,
        atols,
        max_step_size,
        first_step_size,
        force_retain_solver
    );

    // .release() removes ownership from the unique_ptr and yields the raw pointer
    return result.release(); 
}

// The Getters
// These functions allow Numba to peek inside the C++ class
bool cysolver_get_success(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->success;
}

size_t cysolver_get_size(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->size;
}

size_t cysolver_get_num_dy(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->num_dy;
}

double* cysolver_get_t_ptr(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->time_domain_vec.data();
}

double* cysolver_get_y_ptr(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->solution.data();
}

int cysolver_get_status(void* ptr) {
    return static_cast<int>(static_cast<CySolverResult*>(ptr)->status);
}

// The Destructor
// Numba will call this to prevent memory leaks
void cysolver_free(void* ptr) {
    if (ptr != nullptr) {
        delete static_cast<CySolverResult*>(ptr);
    }
}
