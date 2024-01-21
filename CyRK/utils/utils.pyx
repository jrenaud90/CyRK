# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc


cdef inline void* allocate_mem(size_t size, char* var_name):
    cdef void* new_memory = PyMem_Malloc(size)
    if not new_memory:
        raise MemoryError(f'Failed to allocate memory for {var_name}\n\tRequested size = {size}.')
    return new_memory


cdef inline void* reallocate_mem(void* old_pointer, size_t new_size, char* var_name):
    cdef void* new_memory = PyMem_Realloc(old_pointer, new_size)
    if not new_memory:
        raise MemoryError(f'Failed to *re*allocate memory for {var_name}\n\tRequested size = {new_size}.')
    return new_memory
