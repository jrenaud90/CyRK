cdef struct LinkedList:
    size_t size
    double* array_ptr
    LinkedList* next

cdef void* allocate_mem(size_t size, char* var_name) noexcept nogil

cdef void* reallocate_mem(void* old_pointer, size_t new_size, char* var_name) noexcept nogil

cdef void free_mem(void* pointer) noexcept nogil