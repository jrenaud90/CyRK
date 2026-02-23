# Numba `nbsolve_ivp`

:::{attention}
**Deprecation Warning!** As of CyRK v0.17.0 the original `CyRK.nbsolve_ivp` is marked for deprecation. In a future
release it will be replaced by the new `CyRK.nbsolve2_ivp`. These two functions have similar purposes but different
calling methods and very different internals. Details about both can be found on this page. As of CyRK v0.17.0,
the current `nbsolve_ivp` will print out a warning message discussing this deprecation. These messages have a negative
impact on performance and can be disabled by passing `warnings=False` to the solver.
:::

## `CyRK.nbsolve_ivp`
CyRK provides a `numba.njit` safe ODE solver via `from CyRK import nbsolve_ivp`. This solver has limited functionality
compared to `CyRK.pysolve_ivp` and `CyRK.cysolve_ivp`. It uses an entirely independent backend from the other methods.

### Limitations
- Does not support dense outputs.
- Its use of `t_eval` is different and less accurate (see conversation [here](Dense_Output_and_t_eval.md)).
- It does not support events.
- Its output does not follow the `scipy` or CyRK's `CySolution` format. Instead it is a tuple of just:
    - `time_domain` (`np.ndarray`)
    - `y` solution (`np.ndarray`)
    - `success` (`bool`)
    - `message` (`str`)

### Function Signature
`CyRK.nbsolve_ivp` has the following call signature.
```python
@njit(cache=False, fastmath=False)
def nbsolve_ivp(
        diffeq: callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        args: tuple = tuple(),
        rtol: float = 1.e-3,
        atol: float = 1.e-6,
        rtols: np.ndarray = EMPTY_ARR,
        atols: np.ndarray = EMPTY_ARR,
        max_step: float = np.inf,
        first_step: float = None,
        rk_method: int = 1,
        t_eval: np.ndarray = EMPTY_ARR,
        capture_extra: bool = False,
        interpolate_extra: bool = False,
        max_num_steps: int = 0,
        warnings: bool = True
        ):
    """ A Numba-safe Runge-Kutta Integrator based on Scipy's solve_ivp RK integrator.

    Parameters
    ----------
    diffeq : callable
        An njit-compiled function that defines the derivatives of the problem.
    t_span : Tuple[float, float]
        A tuple of the beginning and end of the integration domain's dependent variables.
    y0 : np.ndarray
        1D array of the initial values of the problem at t_span[0]
    args : tuple = tuple()
        Any additional arguments that are passed to dffeq.
    rtol : float = 1.e-3
        Integration relative tolerance used to determine optimal step size.
    atol : float = 1.e-6
        Integration absolute tolerance used to determine optimal step size.
    max_step : float = np.inf
        Maximum allowed step size.
    first_step : float = None
        Initial step size. If `None`, then the function will attempt to determine an appropriate initial step.
    rk_method : int = 1
        The type of RK method used for integration
            0 = RK23
            1 = RK45
            2 = DOP853
    t_eval : np.ndarray = None
        If provided, then the function will interpolate the integration results to provide them at the
            requested t-steps.
    capture_extra : bool = False
        If True, then extra output will be captured from the differential equation.
        See CyRK's Documentation/Extra Output.md for more information
    interpolate_extra : bool = False
        If True, then extra output will be interpolated (along with y) at t_eval. Otherwise, y will be interpolated
         and then differential equation will be called to find the output at each t in t_eval.
    max_num_steps : int = 0
        Maximum number of steps integrator is allowed to take.
        If set to 0 (the default) then an infinite number of steps are allowed.
    warnings : bool = True
        If True, then warnings will be raised which can slow down integration.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.
    .. [2] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.

    Returns
    -------
    time_domain : np.ndarray
        The final time domain. This is equal to t_eval if it was provided.
    y_results : np.ndarray
        The solution of the differential equation provided for each time_result.
            If `capture_extra` was set to True then this will output both y and any extra parameters calculated by the
             differential equation. The format of this output will look like:
            
            y_results[0:y_size, :]          = ... # Actual y-results calculated by the diffeq solver
            y_results[y_size:extra_size, :] = ... # Extra outputs captured alongside y during integration

    success : bool
        Final integration success flag.
    message : str
        Any integration messages, useful if success=False.

    """
```


## `CyRK.nbsolve2_ivp`
**`CyRK.nbsolve2_ivp` is currently considered experimental! Please perform testing before using in any production
environment. Also expect syntax and functionality to change rapidly.**

As of CyRK v0.17.0, a new numba-safe function, `nbsolve2_ivp` was introduced. This function does use the same C++ 
backend as `cysolve_ivp` and `pysolve_ivp`. It uses ctypes to build C safe types that are sent to `cysolve_ivp`. The
result of integration is a C++ class `CySolverResult`. However, numba does not currently support wrapping C++ classes
directly (follow this [issue](https://github.com/numba/numba/issues/9803) if interested). Instead the result's raw
pointer is stored in a numba [jitclass](https://numba.readthedocs.io/en/stable/user/jitclass.html) called
`NbCySolverResult`. This jitclass uses light C getters to retrieve information from the underlying C++ class instance.
The class can be used in both Python and njit'd functions.

While there are a number of downsides of this approach (see limitations below), the advantage is that all of `CyRK`'s
solvers will soon use the same backend. Meaning all functionality, new integrators, bug fixes, performance improvements,
etc. will only have to be implemented once and then be available to all solvers. 

_For this reason, `nbsolve2_ivp` will replace `nbsolve_ivp` in a future release!_

### Limitations and Important Considerations
There are several limitations and considerations when using `nbsolve2_ivp`. 

:::{important}
The `NbCySolverResult` result releases the `CySolverResult` unique pointer so its memory is no longer
  managed by the C++ backend. Currently, jitclasses do not allow for `\_\_del\_\_` methods so we can not automatically 
  release the `CySolverResult` memory when the class is deleted or goes out of scope (this is being actively worked on
  see this [issue](https://github.com/numba/numba/issues/8470) and this [pr](https://github.com/numba/numba/pull/10383)).
  Users must manually call `solution.free()` on the returned solution from `nbsolve2_ivp`. Otherwise a memory leak
  will occur.
:::

- `nbsolve2_ivp` does not currently support [events](Events.md).
- Additional args provided to the diffeq function must be double floating point numbers. And they must be provided to 
  `nbsolve2_ivp` as a numpy array (even if it is only size 1).
- The differential equation must have the following format exactly:
  `def f(dy: double[::1], t: double, y: double[::1], args: double[::1])`
  It also must be compatible with `numba.njit`.
- The diffeq must be compiled to ctypes and its address provided to `nbsolve2_ivp`. CyRK provides helpers to assist with
  this: `from CyRK import cyjit, nb_diffeq_addr`. `cyjit` is the cfunc generator. `nb_diffeq_addr` automatically applies
  `cyjit` and returns the address. Note: this must be called from pure python, not njit'd code!
- jitclass does not support \_\_call\_\_ so when dense output is on you must use `solution.call(...)` or
  `solution.call_vectorize(...)`. `solution()` will not work.

### How to Use
Example usage
```python
from numba import njit

# diffeq must be a njit safe function! You don't have to actually njit it but it won't hurt and it can 
#  help you determine if it will work with njit or not.
@njit
def diffeq(dy, t, y, args):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]

args = np.asarray((0.01, 0.02), dtype=np.float64, order='C')   # Args Must be a numpy.ndarray of np.float64 values!
initial_conds = np.asarray((20., 20.), dtype=np.float64, order='C')
time_span = (0., 10.)

from CyRK import nbsolve2_ivp, nb_diffeq_addr

# We need to compile the diffeq into cfunc. We will use CyRK's nb_diffeq_addr helper.
# Note this must be called from pure Python! It can't be used inside a njit'd function.
diffeq_addr = nb_diffeq_add(diffeq)

result = nbsolve_ivp(
    diffeq_addr,
    time_span,
    initial_conds,
    method='RK45',
    args=args,
    rtol=1.0e-3,
    atol=1.0e-6)

print("Integration Success Status:", result.success)
# result.t
# result.y
# ... etc ...

# IMPORTANT!!! You must manually free the solution memory when you are done with it!
# It will not free when result goes out of scope!!
result.free()
```

### Function Signature
`CyRK.nbsolve2_ivp` has the following call signature.
```python
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
    Numba-compiled entry point for the C++ CyRK ODE solver.

    Parameters
    ----------
    diffeq_address : int
        Memory address of the compiled differential equation C-callback.
    t_span : tuple of float
        Interval of integration (t0, tf).
    y0 : numpy.ndarray
        Initial state vector.
    method : str, optional
        Integration method ('RK45', 'RK23', 'DOP853'). Default is 'RK45'.
    t_eval : numpy.ndarray, optional
        Times at which to store the computed solution.
    dense_output : bool, optional
        Whether to compute a continuous-time polynomial interpolation. Default is False.
    args : numpy.ndarray, optional
        Additional arguments passed to the differential equation.
    rtol, atol : float, optional
        Relative and absolute tolerances.
    rtols, atols : numpy.ndarray, optional
        Vector-valued relative and absolute tolerances.
    num_extra : int, optional
        Number of extra output variables computed by the diffeq.
    expected_size : int, optional
        Estimated number of steps to pre-allocate memory.
    max_step : float, optional
        Maximum allowed step size.
    first_step : float, optional
        Initial step size guess.
    max_num_steps : int, optional
        Maximum number of steps allowed before termination.
    max_ram_MB : int, optional
        Maximum memory allowed for solution storage in Megabytes.
    force_retain_solver : bool, optional
        Whether to keep the solver instance alive after integration.

    Returns
    -------
    NbCySolverResult
        A jitclass object containing the integration results and solution views.
    """
```
### `NbCySolverResult` Solution jitclass
`nbsolve2_ivp` returns a numba [jitclass](https://numba.readthedocs.io/en/stable/user/jitclass.html) that has access
to the underlying memory of the integration solution's `CySolverResult` C++ class. This jitclass can be used in pure
Python or in a numba njit'd function. The following methods and properties are available to users.

```python
```
