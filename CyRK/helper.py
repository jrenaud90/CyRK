""" Helper function to convert conventional solve_ivp or numba diffeq's into a format that cyrk can accept. """
import numpy as np
from numba import njit

def nb2cy(diffeq: callable, use_njit: bool = True, cache_njit: bool = False) -> callable:
    """ Convert numba/scipy differential equation functions to the cyrk format.

    Parameters
    ----------
    diffeq : callable
        Differential equation function.
    use_njit : bool = True
        If True, the final function will be njited.
    cache_njit : bool = False
        If True, then the njit-complied function will be cached.

    Returns
    -------
    diffeq_cyrk : callable
        cyrk-safe differential equation function.
    """


    if use_njit:
        if cache_njit:
            njit_ = njit(cache=True)
        else:
            njit_ = njit(cache=False)
    else:
        def njit_(func):
            return func

    @njit_
    def diffeq_cyrk(t, y, dy, *args):
        # Cython integrator requires the arguments to be passed as input args
        dy_ = diffeq(t, y, *args)

        # Set the input dy items equal to the output
        dy[:] = dy_

    return diffeq_cyrk


def cy2nb(diffeq: callable, use_njit: bool = True, cache_njit: bool = False) -> callable:
    """ Convert cyrk differential equation functions to the numba/scipy format.

    Parameters
    ----------
    diffeq : callable
        Differential equation function.
    use_njit : bool = True
        If True, the final function will be njited.
    cache_njit : bool = False
        If True, then the njit-complied function will be cached.

    Returns
    -------
    diffeq_nbrk : callable
        numba/scipy-safe differential equation function.
    """

    if use_njit:
        if cache_njit:
            njit_ = njit(cache=True)
        else:
            njit_ = njit(cache=False)
    else:
        def njit_(func):
            return func

    @njit_
    def diffeq_nbrk(t, y, *args):
        # Cython integrator requires the arguments to be passed as input args
        dy = np.empty_like(y)
        diffeq(t, y, dy, *args)

        return dy

    return diffeq_nbrk