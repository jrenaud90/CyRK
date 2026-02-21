from typing import Tuple
import ctypes

import numba as nb
from numba import carray
import numpy as np

# Import your newly compiled Cython module
import CyRK.nb.numba_c_api as nb_api
from CyRK.cy.common import MAX_SIZE, DBL_SIZE, CyrkErrorCodes
from CyRK.cy.pyhelpers import get_error_message, find_ode_method_int

# ---------------------------------------------------------
# Define the Function Signatures 
# ctypes.CFUNCTYPE(return_type, arg1_type, arg2_type, ...)
# ---------------------------------------------------------

# Getters (Return types first, then argument types)
bool_func_sig   = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
size_func_sig   = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)
int_func_sig    = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
ptr_func_sig    = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
free_func_sig   = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

# --- Main Solver Signature ---
solve_func_sig = ctypes.CFUNCTYPE(
    ctypes.c_void_p,       # [RETURN] void* (unique_ptr released)
    ctypes.c_void_p,       # [01] DiffeqFuncType diffeq_ptr
    ctypes.c_double,       # [02] double t_start
    ctypes.c_double,       # [03] double t_end
    ctypes.c_void_p,       # [04] double* y0_ptr
    ctypes.c_size_t,       # [05] size_t y0_len
    ctypes.c_int,          # [06] int integration_method_int
    ctypes.c_size_t,       # [07] size_t expected_size
    ctypes.c_size_t,       # [08] size_t num_extra
    ctypes.c_void_p,       # [09] char* args_ptr
    ctypes.c_size_t,       # [10] size_t args_len
    ctypes.c_size_t,       # [11] size_t max_num_steps
    ctypes.c_size_t,       # [12] size_t max_ram_MB
    ctypes.c_bool,         # [13] cpp_bool capture_dense_output
    ctypes.c_void_p,       # [14] double* t_eval_ptr
    ctypes.c_size_t,       # [15] size_t t_eval_len
    ctypes.c_void_p,       # [16] PreEvalFunc pre_eval_func
    ctypes.c_void_p,       # [17] Event* events_ptr
    ctypes.c_size_t,       # [18] size_t events_len
    ctypes.c_void_p,       # [19] double* rtols_ptr
    ctypes.c_size_t,       # [20] size_t rtols_len
    ctypes.c_void_p,       # [21] double* atols_ptr
    ctypes.c_size_t,       # [22] size_t atols_len
    ctypes.c_double,       # [23] double max_step_size
    ctypes.c_double,       # [24] double first_step_size
    ctypes.c_bool          # [25] cpp_bool force_retain_solver
)

# ---------------------------------------------------------
# Instantiate the Callables using the Memory Addresses
# ---------------------------------------------------------
c_get_success = bool_func_sig(nb_api.get_success_func_ptr())
c_get_size    = size_func_sig(nb_api.get_size_func_ptr())
c_get_num_dy  = size_func_sig(nb_api.get_num_dy_func_ptr())
c_get_status  = int_func_sig(nb_api.get_status_func_ptr())
c_get_t_ptr   = ptr_func_sig(nb_api.get_t_func_ptr())
c_get_y_ptr   = ptr_func_sig(nb_api.get_y_func_ptr())
c_free        = free_func_sig(nb_api.get_free_func_ptr())

c_numba_cysolve_ivp = solve_func_sig(nb_api.get_solve_func_ptr())


# ---------------------------------------------------------
# Numba Solution class wrapper
# ---------------------------------------------------------
# Define the types for the jitclass attributes
# ctypes.c_void_p translates to a pointer-sized integer in Numba (intp)
spec = [
    ('_ptr', nb.intp),
]


# ---------------------------------------------------------
# Diffeq function signature
# ---------------------------------------------------------
# The user must wrap their diffeq with this wrapper before passing it to nbsolve_ivp
cyjit_sig = nb.types.void(
    nb.types.CPointer(nb.types.float64),  # dy (double*)
    nb.types.float64,                     # t  (double)
    nb.types.CPointer(nb.types.float64),  # y  (double*)
    nb.types.CPointer(nb.types.float64)   # args (double*)
)
cyjit = nb.cfunc(cyjit_sig)
def nb_diffeq_addr(diffeq_func):
    return cyjit(diffeq_func).address


# ---------------------------------------------------------
# Numba safe storage class
# ---------------------------------------------------------
# Pre-convert some error codes
_ARGUMENT_NOT_SET = np.int32(CyrkErrorCodes.ARGUMENT_NOT_SET)

# TODO as of CyRK v0.17.0 this structure is experimental. 
# There is no __del__ method support for jitclasses so the user must manually call .free() when they are done with it
# Otherwise the pointer to the CySolverResult instance will hang and create a memory leak.
# There is a open PR in numba that is looking to add this functionality. once it is added then I think this will be ready for prime time.
# https://github.com/numba/numba/pull/10383
@nb.experimental.jitclass(spec)
class NbCySolverResult:
    def __init__(self, ptr):
        self._ptr = ptr
    
    @property
    def cyresult_set(self):
        """Checks if the underlying CyResult instance is set or if its been released."""
        return self._ptr != 0

    @property
    def success(self):
        """Did the solver succeed?"""
        if self.cyresult_set:
            return c_get_success(self._ptr)
        else:
            return False
    
    @property
    def status(self):
        """The error/status code of the solver."""
        if self.cyresult_set:
            return c_get_status(self._ptr)
        else:
            return _ARGUMENT_NOT_SET

    @property
    def size(self):
        """Number of steps taken/stored."""
        if self.cyresult_set:
            return c_get_size(self._ptr)
        else:
            return np.uint64(0)

    @property
    def num_dy(self):
        """Number of equations."""
        if self.cyresult_set:
            return c_get_num_dy(self._ptr)
        else:
            return np.uint64(0)

    @property
    def t(self):
        """Zero-copy view of the time domain vector."""
        if self.cyresult_set:
            if self.size == 0:
                return np.empty(0, dtype=np.float64)
                
            t_ptr = c_get_t_ptr(self._ptr)
            # Create a 1D array view of shape (size,)
            return carray(t_ptr, (self.size,))
        else:
            return np.empty(0, dtype=np.float64)

    @property
    def y(self):
        """Zero-copy view of the solution array."""
        if self.cyresult_set:
            if self.size == 0:
                return np.empty((0, 0), dtype=np.float64)
                
            y_ptr = c_get_y_ptr(self._ptr)
            # Based on your Cython reshape logic: .reshape((size, num_dy)).T
            # We can construct the 2D carray and transpose it so it matches WrapCySolverResult
            y_view = carray(y_ptr, (self.size, self.num_dy))
            return y_view.T
        else:
            return np.empty((0, 0), dtype=np.float64)

    def free(self):
        """
        Numba jitclasses do not support __del__. 
        This must be called manually when done with the object to free the C++ heap memory!
        """
        if self.cyresult_set:
            c_free(self._ptr)
            self._ptr = 0  # Null out the pointer so we don't double-free


# ---------------------------------------------------------
# Main njit Safe Solver
# ---------------------------------------------------------
# Create numba safe defaults
EMPTY_ARR = np.empty(0, dtype=np.float64)

# Wrap helper functions
nb_find_ode_method_int = nb.njit(find_ode_method_int)

@nb.njit
def nbsolve2_ivp(
        diffeq_address: nb.intp,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        method: str = 'RK45',
        t_eval: np.ndarray = EMPTY_ARR,
        # Capture dense not currently supported by nbsolve_ivp
        # dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args: np.ndarray = EMPTY_ARR,
        rtol: float = 1.e-3,
        atol: float = 1.e-6,
        rtols: np.ndarray = EMPTY_ARR,
        atols: np.ndarray = EMPTY_ARR,
        num_extra: int = 0,
        expected_size: int = 0,
        max_step_size: float = MAX_SIZE,
        first_step_size: float = 0.0,
        max_num_steps: int = 0,
        max_ram_MB: int = 2000,
        force_retain_solver: bool = True
    ):
    """
    Numba-compiled wrapper for the CyRK C++ ODE solver.
    """

    # Pre-eval functions are not currently supported by CyRK
    pre_eval_func_ptr = 0  # nullptr

    # Capture extra not currently supported by nbsolve_ivp
    capture_dense_output = False

    # Events not currently supported by nbsolve_ivp
    events_ptr = 0  # nullptr
    events_len = 0

    # Parse rtol/atol
    if rtols.size == 0:
        # Only scalar provided; create a new array
        # Don't want to resize the input because it is a global variable and things could get fucky
        rtols = np.array((1,), dtype=np.float64)
        rtols[0] = rtol
    if atols.size == 0:
        atols = np.array((1,), dtype=np.float64)
        atols[0] = atol

    # Parse the integration method
    integration_method_int = nb_find_ode_method_int(method)

    # Parse time tuple
    t_start = t_span[0]
    t_end = t_span[1]

    # Args must be converted to char* to work with CySolver. However, we want to user to work with doubles (only arg type currently supported).
    # We will use zero cost abstraction on the diffeq call by telling numba the address is really a double* even though cysolve sends it a char*.
    # However, we still need to convert the input (which is a double[::1]) to char* to work with cysolve.
    args_bytes_size = args.size * DBL_SIZE
    
    # Call the C-API directly. Numba translates this to a fast C function call.
    # We pass the memory address of the numpy arrays using .ctypes.data
    ptr = c_numba_cysolve_ivp(
        diffeq_address,
        np.float64(t_start),
        np.float64(t_end),
        y0.ctypes.data, np.uint64(y0.size),
        np.int32(integration_method_int),
        np.uint64(expected_size),
        np.uint64(num_extra),
        args.ctypes.data, np.uint64(args_bytes_size),
        np.uint64(max_num_steps),
        np.uint64(max_ram_MB),
        capture_dense_output,
        t_eval.ctypes.data, np.uint64(t_eval.size),
        pre_eval_func_ptr,
        events_ptr, np.uint64(events_len),
        rtols.ctypes.data, np.uint64(rtols.size),
        atols.ctypes.data, np.uint64(atols.size),
        np.float64(max_step_size),
        np.float64(first_step_size),
        force_retain_solver
    )
    
    # Wrap the returned pointer in our Scipy-like jitclass
    sol = NbCySolverResult(ptr)
    
    return sol


def test_nbsolve_ivp(    
        diffeq: callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        method: str = 'RK45',
        t_eval: np.ndarray = EMPTY_ARR,
        # Capture dense not currently supported by nbsolve_ivp
        # dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args: np.ndarray = EMPTY_ARR,
        rtol: float = 1.e-3,
        atol: float = 1.e-6,
        rtols: np.ndarray = EMPTY_ARR,
        atols: np.ndarray = EMPTY_ARR,
        num_extra: int = 0,
        expected_size: int = 0,
        max_step_size: float = MAX_SIZE,
        first_step_size: float = 0.0,
        max_num_steps: int = 0,
        max_ram_MB: int = 2000,
        force_retain_solver: bool = True
    ):

    diffeq_addr = nb_diffeq_addr(diffeq)

    sol = nbsolve2_ivp(
        diffeq_addr,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        # Capture dense not currently supported by nbsolve_ivp
        # dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args=args,
        rtol=rtol,
        atol=atol,
        rtols=rtols,
        atols=atols,
        num_extra=num_extra,
        expected_size=expected_size,
        max_step_size=max_step_size,
        first_step_size=first_step_size,
        max_num_steps=max_num_steps,
        max_ram_MB=max_ram_MB,
        force_retain_solver=force_retain_solver
    )

    # Release the storage
    sol.free()


@nb.njit
def nb_test_nbsolve_ivp(
        diffeq_address: nb.intp,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        method: str = 'RK45',
        t_eval: np.ndarray = EMPTY_ARR,
        # Capture dense not currently supported by nbsolve_ivp
        # dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args: np.ndarray = EMPTY_ARR,
        rtol: float = 1.e-3,
        atol: float = 1.e-6,
        rtols: np.ndarray = EMPTY_ARR,
        atols: np.ndarray = EMPTY_ARR,
        num_extra: int = 0,
        expected_size: int = 0,
        max_step_size: float = MAX_SIZE,
        first_step_size: float = 0.0,
        max_num_steps: int = 0,
        max_ram_MB: int = 2000,
        force_retain_solver: bool = True
    ):

    sol = nbsolve2_ivp(
        diffeq_address,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        # Capture dense not currently supported by nbsolve_ivp
        # dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args=args,
        rtol=rtol,
        atol=atol,
        rtols=rtols,
        atols=atols,
        num_extra=num_extra,
        expected_size=expected_size,
        max_step_size=max_step_size,
        first_step_size=first_step_size,
        max_num_steps=max_num_steps,
        max_ram_MB=max_ram_MB,
        force_retain_solver=force_retain_solver
    )

    # Release the storage
    sol.free()
