cdef extern from "events.hpp" nogil:
    ctypedef double (*EventFunc)(double, char*)