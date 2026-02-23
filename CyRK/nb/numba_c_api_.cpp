// c_numba_bridge.cpp
#include <string>
#include "cysolution.hpp"
#include "cysolve.hpp"
#include "c_common.hpp"

// ============================================================================
// The Execution Wrapper
// ============================================================================
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

// ============================================================================
// CyRK.cy utilities
// ============================================================================
void cysolver_get_status_message_buffer(int status_code, char* buffer, size_t max_len)
{
    CyrkErrorCodes code = static_cast<CyrkErrorCodes>(status_code);
    std::string msg;
    
    // Safely look up the string in the C++ map
    auto it = CyrkErrorMessages.find(code);
    if (it != CyrkErrorMessages.end()) {
        msg = it->second;
    } else {
        msg = "UNKNOWN_ERROR";
    }
    
    // Copy into the Numba-provided buffer
    strncpy(buffer, msg.c_str(), max_len - 1);
    buffer[max_len - 1] = '\0';
}

// ============================================================================
// Core Getters
// ============================================================================
bool cysolver_get_success(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->success;
}

size_t cysolver_get_size(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->size;
}

size_t cysolver_get_num_y(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->num_y;
}

size_t cysolver_get_num_dy(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->num_dy;
}

size_t cysolver_get_steps_taken(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->steps_taken;
}

size_t cysolver_get_num_interpolates(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->num_interpolates;
}

int cysolver_get_status(void* ptr) {
    return static_cast<int>(static_cast<CySolverResult*>(ptr)->status);
}

bool cysolver_get_event_terminated(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->event_terminated;
}

size_t cysolver_get_num_events(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->num_events;
}

// ============================================================================
// Array Pointers
// ============================================================================
double* cysolver_get_t_ptr(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->time_domain_vec.data();
}

double* cysolver_get_y_ptr(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->solution.data();
}

// ============================================================================
// Dense Output Methods
// ============================================================================
int cysolver_call_call(void* ptr, const double t, double* y_interp_ptr) {
    return static_cast<int>(static_cast<CySolverResult*>(ptr)->call(t, y_interp_ptr));
}

int cysolver_call_call_vectorize(void* ptr, const double* t_array_ptr, const size_t len_t, double* y_interp_ptr) {
    return static_cast<int>(static_cast<CySolverResult*>(ptr)->call_vectorize(t_array_ptr, len_t, y_interp_ptr));
}

// ============================================================================
// Diagnostic Getters
// ============================================================================
int cysolver_get_direction(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->direction_flag;
}

bool cysolver_get_capture_extra(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->capture_extra;
}

bool cysolver_get_capture_dense(void* ptr) {
    return static_cast<CySolverResult*>(ptr)->capture_dense_output;
}

int cysolver_get_method(void* ptr) {
    return static_cast<int>(static_cast<CySolverResult*>(ptr)->integrator_method);
}

// ============================================================================
// Args and State Getters
// ============================================================================
size_t cysolver_get_args_size(void* ptr) {
    if (static_cast<CySolverResult*>(ptr)->config_uptr) {
        return static_cast<CySolverResult*>(ptr)->config_uptr->args_vec.size();
    }
    return 0;
}

char* cysolver_get_args_ptr(void* ptr) {
    if (static_cast<CySolverResult*>(ptr)->config_uptr) {
        return static_cast<CySolverResult*>(ptr)->config_uptr->args_vec.data();
    }
    return nullptr;
}

double cysolver_get_t_now(void* ptr) {
    if (static_cast<CySolverResult*>(ptr)->solver_uptr) {
        return static_cast<CySolverResult*>(ptr)->solver_uptr->t_now;
    }
    return 0.0;
}

double* cysolver_get_y_now_ptr(void* ptr) {
    if (static_cast<CySolverResult*>(ptr)->solver_uptr) {
        return static_cast<CySolverResult*>(ptr)->solver_uptr->y_now_ptr;
    }
    return nullptr;
}

double* cysolver_get_dy_now_ptr(void* ptr) {
    if (static_cast<CySolverResult*>(ptr)->solver_uptr) {
        return static_cast<CySolverResult*>(ptr)->solver_uptr->dy_now_ptr;
    }
    return nullptr;
}

// ============================================================================
// String Buffer Copier
// ============================================================================
void cysolver_get_message_buffer(void* ptr, char* buffer, size_t max_len) {
    std::string msg = static_cast<CySolverResult*>(ptr)->message;
    strncpy(buffer, msg.c_str(), max_len - 1);
    buffer[max_len - 1] = '\0'; // Ensure null termination
}

// ============================================================================
// Destructor
// ============================================================================
void cysolver_free(void* ptr) {
    if (ptr != nullptr) {
        delete static_cast<CySolverResult*>(ptr);
    }
}