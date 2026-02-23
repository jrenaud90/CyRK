import math
from typing import Tuple, Optional
import ctypes

import numba as nb
from numba import carray
import numpy as np

# Import your newly compiled Cython module
import CyRK.nb.numba_c_api as nb_api
from CyRK.cy.common import MAX_SIZE, DBL_SIZE, CyrkErrorCodes
from CyRK.cy.pyhelpers import find_ode_method_int

# ---------------------------------------------------------
# Define the Function Signatures 
# ctypes.CFUNCTYPE(return_type, arg1_type, arg2_type, ...)
# ---------------------------------------------------------

# CyRK.cy utilities
status_msg_buf_func_sig = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t)

# Getters (Return types first, then argument types)
bool_func_sig   = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
size_func_sig   = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)
int_func_sig    = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)
dbl_func_sig    = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)
ptr_func_sig    = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
char_ptr_sig    = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_uint8), ctypes.c_void_p)
free_func_sig   = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

# We map all pointers to c_size_t (uint64) for safe Numba integer passing
call_func_sig     = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t)
call_vec_func_sig = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t)
msg_buf_func_sig  = ctypes.CFUNCTYPE(None, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t)

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
    ctypes.c_double,       # [23] double max_step
    ctypes.c_double,       # [24] double first_step
    ctypes.c_bool          # [25] cpp_bool force_retain_solver
)

# ---------------------------------------------------------
# Instantiate the Callables using the Memory Addresses
# ---------------------------------------------------------
c_get_status_msg_buf = status_msg_buf_func_sig(nb_api.get_status_message_buffer_func_ptr())
c_get_success = bool_func_sig(nb_api.get_success_func_ptr())
c_get_size    = size_func_sig(nb_api.get_size_func_ptr())
c_get_num_y   = size_func_sig(nb_api.get_num_y_func_ptr())
c_get_num_dy  = size_func_sig(nb_api.get_num_dy_func_ptr())
c_get_steps   = size_func_sig(nb_api.get_steps_taken_func_ptr())
c_get_interp  = size_func_sig(nb_api.get_num_interpolates_func_ptr())
c_get_status  = int_func_sig(nb_api.get_status_func_ptr())
c_get_t_ptr   = ptr_func_sig(nb_api.get_t_func_ptr())
c_get_y_ptr   = ptr_func_sig(nb_api.get_y_func_ptr())
c_free        = free_func_sig(nb_api.get_free_func_ptr())
c_call        = call_func_sig(nb_api.get_call_call_func_ptr())
c_call_vec    = call_vec_func_sig(nb_api.get_call_call_vectorize_func_ptr())
c_get_direction = int_func_sig(nb_api.get_direction_func_ptr())
c_get_cap_extra = bool_func_sig(nb_api.get_capture_extra_func_ptr())
c_get_cap_dense = bool_func_sig(nb_api.get_capture_dense_func_ptr())
c_get_method    = int_func_sig(nb_api.get_method_func_ptr())
c_get_args_size = size_func_sig(nb_api.get_args_size_func_ptr())
c_get_args_ptr  = char_ptr_sig(nb_api.get_args_ptr_func_ptr())
c_get_t_now     = dbl_func_sig(nb_api.get_t_now_func_ptr())
c_get_y_now     = ptr_func_sig(nb_api.get_y_now_ptr_func_ptr())
c_get_dy_now    = ptr_func_sig(nb_api.get_dy_now_ptr_func_ptr())
c_get_msg_buf   = msg_buf_func_sig(nb_api.get_message_buffer_func_ptr())

c_numba_cysolve_ivp = solve_func_sig(nb_api.get_solve_func_ptr())

# ---------------------------------------------------------
# Helpers for Numba Strings and Formatting
# ---------------------------------------------------------
@nb.njit
def get_status_message_str(status_code):
    """ Numba-safe string extractor for C++ Error Code map. """
    max_len = 256
    buf = np.zeros(max_len, dtype=np.uint8)
    
    # Send the integer to C++, C++ fills the buffer
    c_get_status_msg_buf(np.int32(status_code), np.uint64(buf.ctypes.data), np.uint64(max_len))
    
    # Reconstruct the string in Numba
    s = ""
    for i in range(max_len):
        if buf[i] == 0: break  # Null terminator
        s += chr(buf[i])
    return s

@nb.njit
def to_hex(n):
    """Numba-safe integer to hex string converter."""
    chars = "0123456789abcdef"
    s = ""
    while n > 0:
        s = chars[n % 16] + s
        n = n // 16
    return "0x" + (s if s else "0")

@nb.njit
def get_method_str(method_int):
    """ Maps method ints to strings. """
    if method_int == 1:
        return "RK45"
    if method_int == 2:
        return "RK23"
    if method_int == 3:
        return "DOP853"
    return "UNKNOWN_METHOD"

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

# Define the types for the jitclass attributes
# ctypes.c_void_p translates to a pointer-sized integer in Numba (intp)
spec = [
    ('_ptr', nb.intp),
]

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
        return self._ptr != 0

    @property
    def success(self):
        if self.cyresult_set:
            return c_get_success(self._ptr)
        return False
    
    @property
    def status(self):
        if self.cyresult_set:
            return c_get_status(self._ptr)
        return _ARGUMENT_NOT_SET

    @property
    def error_code(self):
        return self.status

    @property
    def size(self):
        if self.cyresult_set:
            return c_get_size(self._ptr)
        return np.uint64(0)

    @property
    def steps_taken(self):
        if self.cyresult_set:
            return c_get_steps(self._ptr)
        return np.uint64(0)

    @property
    def num_y(self):
        if self.cyresult_set:
            return c_get_num_y(self._ptr)
        return np.uint64(0)

    @property
    def num_dy(self):
        if self.cyresult_set:
            return c_get_num_dy(self._ptr)
        return np.uint64(0)

    @property
    def num_interpolates(self):
        if self.cyresult_set:
            return c_get_interp(self._ptr)
        return np.uint64(0)

    @property
    def t(self):
        if self.cyresult_set and self.size > 0:
            return carray(c_get_t_ptr(self._ptr), (self.size,))
        return np.empty(0, dtype=np.float64)

    @property
    def y(self):
        if self.cyresult_set and self.size > 0:
            y_view = carray(c_get_y_ptr(self._ptr), (self.size, self.num_dy))
            return y_view.T
        return np.empty((0, 0), dtype=np.float64)

    def call(self, t):
        """ Evaluate the dense output interpolator at a single float t. """
        if not self.cyresult_set:
            return np.empty((0, 0), dtype=np.float64)
            
        y_interp = np.empty(np.int64(self.num_dy), dtype=np.float64)
        c_call(self._ptr, np.float64(t), np.uint64(y_interp.ctypes.data))
        
        return y_interp.reshape((np.int64(self.num_dy), 1))

    def call_vectorize(self, t_array):
        """ Evaluate the dense output interpolator across a 1D numpy array of times. """
        if not self.cyresult_set:
            return np.empty((0, 0), dtype=np.float64)
            
        len_t = t_array.size
        y_interp = np.empty(np.int64(self.num_dy * len_t), dtype=np.float64)
        
        c_call_vec(
            self._ptr, 
            np.uint64(t_array.ctypes.data), 
            np.uint64(len_t), 
            np.uint64(y_interp.ctypes.data)
        )
        
        return y_interp.reshape((np.int64(len_t), np.int64(self.num_dy))).T

    @property
    def status_message(self):
        """ Returns the C++ error message string for the current status. """
        if not self.cyresult_set:
            return "NULL_POINTER"
            
        # Call the new buffer-based extractor we just built
        return get_status_message_str(self.status)
        
    @property
    def message(self):
        """ Retrieves the C++ std::string by copying bytes to a Numba array. """
        if not self.cyresult_set:
            return ""
        
        # Pre-allocate a 256 byte buffer for the string
        max_len = 256
        buf = np.zeros(max_len, dtype=np.uint8)
        
        # Have C++ copy the string characters into our numpy buffer
        c_get_msg_buf(np.uint64(self._ptr), np.uint64(buf.ctypes.data), np.uint64(max_len))
        
        # Reconstruct the string in Numba
        s = ""
        for i in range(max_len):
            if buf[i] == 0: break  # Null terminator
            s += chr(buf[i])
        return s

    def print_diagnostics(self):
        if not self.cyresult_set:
            print("ERROR: `NbCySolverResult::print_diagnostics` - CySolverResult is Null.")
            return
            
        direction_str = 'Forward' if c_get_direction(self._ptr) == 1 else 'Backward'
        method_str = get_method_str(c_get_method(self._ptr))
        
        diagnostic_str = '----------------------------------------------------\n'
        diagnostic_str += 'CyRK (Numba) - NbCySolverResult Diagnostic.\n'
        diagnostic_str += '----------------------------------------------------\n'
        diagnostic_str += f'# of y:             {self.num_y}.\n'
        diagnostic_str += f'# of dy:            {self.num_dy}.\n'
        diagnostic_str += f'Success:            {self.success}.\n'
        diagnostic_str += f'Error Code:         {self.error_code}.\n'
        diagnostic_str += f'Status:             {self.status_message}.\n'
        diagnostic_str += f'Size:               {self.size}.\n'
        diagnostic_str += f'Steps Taken:        {self.steps_taken}.\n'
        diagnostic_str += f'Integrator Message:\n\t{self.message}\n'
        diagnostic_str += '\n----------------- CySolverResult -------------------\n'
        diagnostic_str += f'Capture Extra:          {c_get_cap_extra(self._ptr)}.\n'
        diagnostic_str += f'Capture Dense Output:   {c_get_cap_dense(self._ptr)}.\n'
        diagnostic_str += f'Integration Direction:  {direction_str}.\n'
        diagnostic_str += f'Integration Method:     {method_str}.\n'
        diagnostic_str += f'# of Interpolates:      {self.num_interpolates}.\n'

        diagnostic_str += '\n---- Additional Argument Info ----\n'
        args_size = c_get_args_size(self._ptr)
        args_size_dbls = np.int64(math.floor(args_size / DBL_SIZE))
        
        diagnostic_str += f'args size (bytes):   {args_size}.\n'
        diagnostic_str += f'args size (doubles): {args_size_dbls}.\n'
        
        args_ptr = c_get_args_ptr(self._ptr)
        
        # Numba hack to cleanly read char* as bytes and float64 at the same time
        if args_ptr == 0:
            diagnostic_str += 'Args Pointer is Null.\n'
        elif args_size > 0:
            # View memory as bytes
            args_bytes = carray(args_ptr, (args_size,))
            
            # Since args_ptr is c_uint8*, cast to c_double* to read floats
            dbl_ptr = ctypes.cast(args_ptr, ctypes.POINTER(ctypes.c_double))
            args_dbls = carray(dbl_ptr, (args_size_dbls,))
            
            dbl_i = 0
            for i in range(args_size):
                if i % 8 == 0:
                    diagnostic_str += f'\n{to_hex(args_bytes[i])}'
                elif i % 8 == 3:
                    diagnostic_str += f' {to_hex(args_bytes[i])}\n'
                elif i % 8 == 7:
                    diagnostic_str += f' {to_hex(args_bytes[i])}\n'
                    diagnostic_str += f'As Double: {args_dbls[dbl_i]:0.5e}\n'
                    dbl_i += 1
                else:
                    diagnostic_str += f' {to_hex(args_bytes[i])}'
        diagnostic_str += 'End of Additional Argument Info.\n'

        # Current State Info
        num_y = self.num_y
        num_dy = self.num_dy
        
        diagnostic_str += '\n------------------ CySolverBase --------------------\n'
        diagnostic_str += f'Integration Method: {method_str}.\n'
        diagnostic_str += f'# of y:             {num_y}.\n'
        diagnostic_str += f'# of dy:            {num_dy}.\n'
        diagnostic_str += '---- Current State Info ----\n'
        
        t_now = c_get_t_now(self._ptr)
        diagnostic_str += f't_now: {t_now}.\n'
        
        y_now_ptr = c_get_y_now(self._ptr)
        if y_now_ptr != 0:
            diagnostic_str += 'y_now:\n'
            y_arr = carray(y_now_ptr, (num_y,))
            for i in range(num_y):
                diagnostic_str += f'\ty{i}  = {y_arr[i]:0.5e}.\n'
                
        dy_now_ptr = c_get_dy_now(self._ptr)
        if dy_now_ptr != 0:
            diagnostic_str += 'dy_now:\n'
            dy_arr = carray(dy_now_ptr, (num_dy,))
            for i in range(num_dy):
                diagnostic_str += f'\tdy{i} = {dy_arr[i]:0.5e}.\n'
                
        diagnostic_str += 'End of Current State Info.\n'
        diagnostic_str += '\n-------------- Diagnostic Complete -----------------\n'
        print(diagnostic_str)

    def free(self):
        if self.cyresult_set:
            c_free(self._ptr)
            self._ptr = 0


# ---------------------------------------------------------
# Main njit Safe Solver
# ---------------------------------------------------------
# Wrap helper functions
nb_find_ode_method_int = nb.njit(find_ode_method_int)

@nb.njit
def nbsolve2_ivp(
        diffeq_address: nb.intp,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        method: str = 'RK45',
        t_eval: Optional[np.ndarray] = None,
        dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args: Optional[np.ndarray] = None,
        rtol: float = 1.e-3,
        atol: float = 1.e-6,
        rtols: Optional[np.ndarray] = None,
        atols: Optional[np.ndarray] = None,
        num_extra: int = 0,
        expected_size: int = 0,
        max_step: float = MAX_SIZE,
        first_step: float = 0.0,
        max_num_steps: int = 0,
        max_ram_MB: int = 2000,
        force_retain_solver: bool = True
    ):
    """
    Numba-compiled wrapper for the CyRK C++ ODE solver.
    """

    # Pre-eval functions are not currently supported by CyRK
    pre_eval_func_ptr = 0  # nullptr

    # Events not currently supported by nbsolve_ivp
    events_ptr = 0  # nullptr
    events_len = 0

    # Parse rtol/atol
    if rtols is None:
        _rtols = np.empty(1, dtype=np.float64)
        _rtols[0] = rtol
    elif rtols.size == 0:
        _rtols = np.empty(1, dtype=np.float64)
        _rtols[0] = rtol
    else:
        _rtols = rtols

    if atols is None:
        _atols = np.empty(1, dtype=np.float64)
        _atols[0] = atol
    elif atols.size == 0:
        _atols = np.empty(1, dtype=np.float64)
        _atols[0] = atol
    else:
        _atols = atols

    # Parse the integration method
    integration_method_int = nb_find_ode_method_int(method)

    # Parse time tuple
    t_start = t_span[0]
    t_end = t_span[1]

    # Parse t_eval
    if t_eval is None:
        _t_eval = np.empty(0, dtype=np.float64)
    else:
        _t_eval = t_eval

    # Parse args
    if args is None:
        _args = np.empty(0, dtype=np.float64)
    else:
        _args = args
    
    # Args must be converted to char* to work with CySolver. However, we want to user to work with doubles (only arg type currently supported).
    # We will use zero cost abstraction on the diffeq call by telling numba the address is really a double* even though cysolve sends it a char*.
    # However, we still need to convert the input (which is a double[::1]) to char* to work with cysolve.
    args_bytes_size = _args.size * DBL_SIZE
    
    # Call the C-API directly. Numba translates this to a fast C function call.
    # We pass the memory address of the numpy arrays using .ctypes.data
    ptr = c_numba_cysolve_ivp(
        np.uint64(diffeq_address),
        np.float64(t_start),
        np.float64(t_end),
        y0.ctypes.data, np.uint64(y0.size),
        np.int32(integration_method_int),
        np.uint64(expected_size),
        np.uint64(num_extra),
        _args.ctypes.data, np.uint64(args_bytes_size),
        np.uint64(max_num_steps),
        np.uint64(max_ram_MB),
        dense_output,
        _t_eval.ctypes.data, np.uint64(_t_eval.size),
        pre_eval_func_ptr,
        events_ptr, np.uint64(events_len),
        _rtols.ctypes.data, np.uint64(_rtols.size),
        _atols.ctypes.data, np.uint64(_atols.size),
        np.float64(max_step),
        np.float64(first_step),
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
        t_eval: np.ndarray = None,
        dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args: np.ndarray = None,
        rtol: float = 1.e-3,
        atol: float = 1.e-6,
        rtols: np.ndarray = None,
        atols: np.ndarray = None,
        num_extra: int = 0,
        expected_size: int = 0,
        max_step: float = MAX_SIZE,
        first_step: float = 0.0,
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
        dense_output=dense_output,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args=args,
        rtol=rtol,
        atol=atol,
        rtols=rtols,
        atols=atols,
        num_extra=num_extra,
        expected_size=expected_size,
        max_step=max_step,
        first_step=first_step,
        max_num_steps=max_num_steps,
        max_ram_MB=max_ram_MB,
        force_retain_solver=force_retain_solver
    )

    # Release the storage
    sol.free()


@nb.njit
def njit_test_nbsolve_ivp(
        diffeq_address: nb.intp,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        method: str = 'RK45',
        t_eval: Optional[np.ndarray] = None,
        dense_output: bool = False,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args: Optional[np.ndarray] = None,
        rtol: float = 1.e-3,
        atol: float = 1.e-6,
        rtols: Optional[np.ndarray] = None,
        atols: Optional[np.ndarray] = None,
        num_extra: int = 0,
        expected_size: int = 0,
        max_step: float = MAX_SIZE,
        first_step: float = 0.0,
        max_num_steps: int = 0,
        max_ram_MB: int = 2000,
        force_retain_solver: bool = True
    ):

    sol = nbsolve2_ivp(
        diffeq_address,
        t_span,
        y0,
        method,
        t_eval,
        dense_output,
        # Events not currently supported by nbsolve_ivp
        # events: Tuple[func]
        args,
        rtol,
        atol,
        rtols,
        atols,
        num_extra,
        expected_size,
        max_step,
        first_step,
        max_num_steps,
        max_ram_MB,
        force_retain_solver
    )

    # Release the storage
    sol.free()
