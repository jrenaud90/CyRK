# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
from libc.stdlib cimport malloc, realloc, free, exit, EXIT_FAILURE
from libc.stdio cimport printf


cdef inline void* allocate_mem(size_t size, char* var_name) noexcept nogil:
    cdef void* new_memory = malloc(size)
    if not new_memory:
        printf('Memory Error: Failed to allocate memory for %s \n\tRequested size = %zu.', var_name, size)
        exit(EXIT_FAILURE)
    return new_memory


cdef inline void* reallocate_mem(void* old_pointer, size_t new_size, char* var_name) noexcept nogil:
    cdef void* new_memory = realloc(old_pointer, new_size)
    if not new_memory:
        printf('Failed to *re*allocate memory for %s \n\tRequested size = %zu.', var_name, new_size)
        exit(EXIT_FAILURE)
    return new_memory

cdef inline void free_mem(void* pointer) noexcept nogil:
    free(pointer)
