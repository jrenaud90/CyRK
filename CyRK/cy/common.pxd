
ctypedef fused double_numeric:
    double
    double complex

cdef double SAFETY
cdef double MIN_FACTOR
cdef double MAX_FACTOR
cdef double MAX_STEP
cdef double INF
cdef double EPS
cdef double EPS_10
cdef double EPS_100
cdef size_t MAX_INT_SIZE
cdef size_t MAX_SIZET_SIZE
cdef double CPU_CACHE_SIZE
cdef double EXPECTED_SIZE_DBL
cdef double EXPECTED_SIZE_DBLCMPLX
cdef double MAX_ARRAY_PREALLOCATE_SIZE_DBL
cdef double MAX_ARRAY_PREALLOCATE_SIZE_DBLCMPLX
cdef double MIN_ARRAY_PREALLOCATE_SIZE
cdef double ARRAY_PREALLOC_TABS_SCALE
cdef double ARRAY_PREALLOC_RTOL_SCALE
cdef size_t RAM_BUFFER_SIZE

cdef void interpolate(
        double* time_domain_full,
        double* time_domain_reduced,
        double_numeric* target_array_full,
        double_numeric* target_array_reduced,
        size_t t_len_full,
        size_t t_len_reduced,
        size_t target_len,
        bint is_complex
        ) noexcept

cdef size_t find_expected_size(
        size_t y_size,
        size_t num_extra,
        double t_delta_abs,
        double rtol_min,
        bint capture_extra,
        bint is_complex
        ) noexcept nogil


cdef void find_max_num_steps(
        size_t y_size,
        size_t num_extra,
        size_t max_num_steps,
        size_t max_ram_MB,
        bint capture_extra,
        bint is_complex,
        bint* user_provided_max_num_steps,
        size_t* max_num_steps_touse) noexcept nogil