# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.stdint cimport uintptr_t

# Expose the raw memory addresses to Python ---

def get_solve_func_ptr():
    return <uintptr_t>&numba_cysolve_ivp

def get_success_func_ptr():
    return <uintptr_t>&cysolver_get_success

def get_size_func_ptr():
    return <uintptr_t>&cysolver_get_size

def get_num_dy_func_ptr():
    return <uintptr_t>&cysolver_get_num_dy

def get_t_func_ptr():
    return <uintptr_t>&cysolver_get_t_ptr

def get_y_func_ptr():
    return <uintptr_t>&cysolver_get_y_ptr

def get_status_func_ptr():
    return <uintptr_t>&cysolver_get_status

def get_free_func_ptr():
    return <uintptr_t>&cysolver_free
