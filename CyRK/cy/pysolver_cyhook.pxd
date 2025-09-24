ctypedef public api void (*DiffeqMethod)(object py_instance) noexcept

ctypedef public api double (*PyEventMethod)(object py_instance, size_t event_index, double t, double* y_ptr) noexcept
