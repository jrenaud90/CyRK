# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
MAX_SIZE = MAX_SIZET_SIZE

# Expose some C/C++ constants to python
DBL_SIZE = sizeof(double)


def get_error_message(int error_code):
    """Return the message for a CyRK status or error code.

    Parameters
    ----------
    error_code : int
        A `CyrkErrorCodes` member or its integer value.

    Returns
    -------
    str
        The code's message, or a generic "unknown error code" message if CyRK defines none for it.
    """
    return c_get_error_message(<CyrkErrorCodes>error_code).decode('utf-8')
