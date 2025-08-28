# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

# The "public api" prefix tells Cython to produce header files "pysolver_api.h" which can be included in C++ files.

cdef public api void call_diffeq_from_cython(object py_instance, DiffeqMethod diffeq):
    """Callback function used by the C++ model to call user-provided python diffeq functions.
    """

    # Call the python diffeq.
    diffeq(py_instance)

cdef public api double call_pyevent_from_cython(object py_instance, PyEventMethod pyevent_method, size_t event_index, double t, double* y_ptr):
    """Callback function used by the C++ model to call user-provided python event functions.
    """

    # Call the python pyevent function.
    return pyevent_method(py_instance, event_index, t, y_ptr)
