# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.stdint cimport uintptr_t

def get_status_message_buffer_func_ptr():
    return <uintptr_t>&cysolver_get_status_message_buffer

# Expose the raw memory addresses to Python ---
def get_solve_func_ptr():
    return <uintptr_t>&numba_cysolve_ivp

def get_success_func_ptr():
    return <uintptr_t>&cysolver_get_success

def get_size_func_ptr():
    return <uintptr_t>&cysolver_get_size

def get_num_y_func_ptr():
    return <uintptr_t>&cysolver_get_num_y

def get_num_dy_func_ptr():
    return <uintptr_t>&cysolver_get_num_dy

def get_steps_taken_func_ptr():
    return <uintptr_t>&cysolver_get_steps_taken

def get_num_interpolates_func_ptr():
    return <uintptr_t>&cysolver_get_num_interpolates

def get_t_func_ptr():
    return <uintptr_t>&cysolver_get_t_ptr

def get_y_func_ptr():
    return <uintptr_t>&cysolver_get_y_ptr

def get_status_func_ptr():
    return <uintptr_t>&cysolver_get_status

def get_call_call_func_ptr():
    return <uintptr_t>&cysolver_call_call

def get_call_call_vectorize_func_ptr():
    return <uintptr_t>&cysolver_call_call_vectorize

def get_direction_func_ptr():
    return <uintptr_t>&cysolver_get_direction

def get_capture_extra_func_ptr():
    return <uintptr_t>&cysolver_get_capture_extra

def get_capture_dense_func_ptr():
    return <uintptr_t>&cysolver_get_capture_dense

def get_method_func_ptr():
    return <uintptr_t>&cysolver_get_method

def get_args_size_func_ptr():
    return <uintptr_t>&cysolver_get_args_size

def get_args_ptr_func_ptr():
    return <uintptr_t>&cysolver_get_args_ptr

def get_t_now_func_ptr():
    return <uintptr_t>&cysolver_get_t_now

def get_y_now_ptr_func_ptr():
    return <uintptr_t>&cysolver_get_y_now_ptr

def get_dy_now_ptr_func_ptr():
    return <uintptr_t>&cysolver_get_dy_now_ptr

def get_message_buffer_func_ptr():
    return <uintptr_t>&cysolver_get_message_buffer

def get_free_func_ptr():
    return <uintptr_t>&cysolver_free