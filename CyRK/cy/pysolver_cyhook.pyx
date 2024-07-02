# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cdef public api int call_diffeq_from_cython(object py_instance):
    """Callback function used by the C++ model.
    The "public api" prefix tells Cython to produce header files "pysolver_api.h" which can be included in
    C++ files.
    """

    return py_instance.diffeq()
